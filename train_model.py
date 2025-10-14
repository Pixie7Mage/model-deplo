"""
Training Script for CRISPR-BERT Model
Trains on I1, I2, and II4 datasets with early stopping
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sequence_encoder import encode_for_cnn, encode_for_bert
from data_loader import load_dataset
from sklearn.model_selection import train_test_split
import os


def load_multiple_datasets(file_paths):
    """
    Load and combine multiple datasets.
    
    Args:
        file_paths: List of paths to dataset files
    
    Returns:
        Combined sequences and labels
    """
    all_sgrna = []
    all_dna = []
    all_labels = []
    
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        sgrna_list, dna_list, labels = load_dataset(file_path)
        all_sgrna.extend(sgrna_list)
        all_dna.extend(dna_list)
        all_labels.extend(labels)
        print(f"  Loaded {len(sgrna_list)} samples")
    
    print(f"\nTotal samples: {len(all_sgrna)}")
    return all_sgrna, all_dna, np.array(all_labels)


def encode_datasets(sgrna_list, dna_list):
    """
    Encode sequences for CNN and BERT inputs.
    
    Returns:
        cnn_inputs, token_ids, segment_ids, position_ids
    """
    print("\nEncoding sequences...")
    batch_size = len(sgrna_list)
    
    # CNN inputs: (batch, 26, 7)
    cnn_inputs = np.array([encode_for_cnn(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    
    # BERT inputs
    token_ids = np.array([encode_for_bert(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    segment_ids = np.zeros((batch_size, 26), dtype=np.int32)
    position_ids = np.tile(np.arange(26), (batch_size, 1))
    
    print(f"CNN input shape: {cnn_inputs.shape}")
    print(f"Token IDs shape: {token_ids.shape}")
    
    return cnn_inputs, token_ids, segment_ids, position_ids


def build_inception_cnn_branch():
    """
    Build the Inception-based CNN branch.
    Input: (26, 7) -> Output: (26, 80)
    """
    inputs = layers.Input(shape=(26, 7), name='cnn_input')
    
    # Inception module with multiple filter sizes
    # 1x1 conv
    conv1x1 = layers.Conv1D(20, 1, padding='same', activation='relu')(inputs)
    
    # 3x1 conv
    conv3x1 = layers.Conv1D(20, 3, padding='same', activation='relu')(inputs)
    
    # 5x1 conv
    conv5x1 = layers.Conv1D(20, 5, padding='same', activation='relu')(inputs)
    
    # Max pooling + 1x1 conv
    maxpool = layers.MaxPooling1D(3, strides=1, padding='same')(inputs)
    pool_conv = layers.Conv1D(20, 1, padding='same', activation='relu')(maxpool)
    
    # Concatenate all branches: 20+20+20+20 = 80
    concat = layers.Concatenate(axis=-1)([conv1x1, conv3x1, conv5x1, pool_conv])
    
    model = models.Model(inputs=inputs, outputs=concat, name='cnn_branch')
    return model


def build_bert_branch(vocab_size=28, embed_dim=256, num_heads=4, num_layers=2, ff_dim=1024):
    """
    Build the BERT branch using Transformer layers.
    Input: token_ids (26,) -> Output: (26, 80)
    """
    # Inputs
    token_input = layers.Input(shape=(26,), dtype=tf.int32, name='token_ids')
    segment_input = layers.Input(shape=(26,), dtype=tf.int32, name='segment_ids')
    position_input = layers.Input(shape=(26,), dtype=tf.int32, name='position_ids')
    
    # Embeddings
    token_embedding = layers.Embedding(vocab_size, embed_dim)(token_input)
    segment_embedding = layers.Embedding(2, embed_dim)(segment_input)
    position_embedding = layers.Embedding(26, embed_dim)(position_input)
    
    # Combine embeddings
    x = token_embedding + segment_embedding + position_embedding
    
    # Transformer layers
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        ffn_output = ffn(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Project to output dimension (80)
    output = layers.Dense(80, activation='relu')(x)
    
    model = models.Model(
        inputs=[token_input, segment_input, position_input],
        outputs=output,
        name='bert_branch'
    )
    return model


def build_crispr_bert_model():
    """
    Build the complete CRISPR-BERT model.
    
    Architecture:
        - CNN Branch: (26, 7) -> (26, 80)
        - BERT Branch: token_ids -> (26, 80)
        - Separate BiGRU layers for each branch
        - Weighted combination (0.2 for CNN, 0.8 for BERT)
        - Dense layers: 80 -> 128 -> 64 -> 2
    """
    # Define inputs
    cnn_input = layers.Input(shape=(26, 7), name='cnn_input')
    token_input = layers.Input(shape=(26,), dtype=tf.int32, name='token_ids')
    segment_input = layers.Input(shape=(26,), dtype=tf.int32, name='segment_ids')
    position_input = layers.Input(shape=(26,), dtype=tf.int32, name='position_ids')
    
    # Build branches
    cnn_branch = build_inception_cnn_branch()
    bert_branch = build_bert_branch()
    
    # Get branch outputs
    cnn_features = cnn_branch(cnn_input)  # (batch, 26, 80)
    bert_features = bert_branch([token_input, segment_input, position_input])  # (batch, 26, 80)
    
    # Separate BiGRU layers (20 units each, bidirectional = 40 output)
    cnn_bigru = layers.Bidirectional(
        layers.GRU(20, return_sequences=True),
        name='cnn_bigru'
    )(cnn_features)  # (batch, 26, 40)
    
    bert_bigru = layers.Bidirectional(
        layers.GRU(20, return_sequences=True),
        name='bert_bigru'
    )(bert_features)  # (batch, 26, 40)
    
    # Take last timestep
    cnn_last = layers.Lambda(lambda x: x[:, -1, :])(cnn_bigru)  # (batch, 40)
    bert_last = layers.Lambda(lambda x: x[:, -1, :])(bert_bigru)  # (batch, 40)
    
    # Apply weights: 0.2 for CNN, 0.8 for BERT
    cnn_weighted = layers.Lambda(lambda x: x * 0.2)(cnn_last)
    bert_weighted = layers.Lambda(lambda x: x * 0.8)(bert_last)
    
    # Concatenate
    combined = layers.Concatenate()([cnn_weighted, bert_weighted])  # (batch, 80)
    
    # Dense layers with dropout
    x = layers.Dropout(0.35)(combined)
    x = layers.Dense(128, activation='relu')(x)
    
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    output = layers.Dense(2, activation='softmax', name='output')(x)
    
    # Build model
    model = models.Model(
        inputs=[cnn_input, token_input, segment_input, position_input],
        outputs=output,
        name='CRISPR_BERT'
    )
    
    return model


def train_crispr_bert(datasets=['datasets/sam.txt'],
                      epochs=100,
                      batch_size=32,
                      patience=5):
    """
    Train CRISPR-BERT model on dataset.
    
    Args:
        datasets: List of dataset file paths
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
    """
    print("=" * 60)
    print("CRISPR-BERT Training")
    print("=" * 60)
    
    # Load datasets
    print("\n[1/5] Loading dataset...")
    sgrna_list, dna_list, labels = load_multiple_datasets(datasets)
    
    # Encode sequences
    print("\n[2/5] Encoding sequences...")
    cnn_inputs, token_ids, segment_ids, position_ids = encode_datasets(sgrna_list, dna_list)
    
    # Split into train/validation
    print("\n[3/5] Splitting data...")
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    X_train = {
        'cnn_input': cnn_inputs[train_idx],
        'token_ids': token_ids[train_idx],
        'segment_ids': segment_ids[train_idx],
        'position_ids': position_ids[train_idx]
    }
    X_val = {
        'cnn_input': cnn_inputs[val_idx],
        'token_ids': token_ids[val_idx],
        'segment_ids': segment_ids[val_idx],
        'position_ids': position_ids[val_idx]
    }
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
    
    # Build model
    print("\n[4/5] Building model...")
    model = build_crispr_bert_model()
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'weight/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create weight directory if it doesn't exist
    os.makedirs('weight', exist_ok=True)
    
    # Train model
    print(f"\n[5/5] Training for {epochs} epochs with early stopping (patience={patience})...")
    print("=" * 60)
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    
    # Save final model
    model.save('weight/final_model.h5')
    print("\nModel saved to 'weight/final_model.h5'")
    
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_crispr_bert(
        datasets=['datasets/sam.txt'],
        epochs=100,
        batch_size=32,
        patience=5
    )
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)
