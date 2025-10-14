"""
CRISPR-BERT Model Inference
Loads trained Keras models and performs predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sequence_encoder import encode_for_cnn, encode_for_bert
from data_loader import load_dataset
import os
import json

# ========== OPTION 1: Single Prediction ==========

def load_trained_model(model_path='weight/final_model.keras'):
    """
    Load a trained Keras model.
    
    Args:
        model_path: Path to the saved Keras model
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            f"Please train the model first using train_model.py"
        )
    
    print(f"Loading model from '{model_path}'...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def load_threshold(threshold_path='weight/threshold_schedule.json'):
    """
    Load adaptive threshold from training.
    
    Args:
        threshold_path: Path to threshold schedule JSON
    
    Returns:
        Optimal threshold value (default 0.5 if not found)
    """
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            threshold = data.get('final_threshold', 0.5)
            print(f"Using adaptive threshold: {threshold:.3f}")
            return threshold
    else:
        print("Using default threshold: 0.5")
        return 0.5


def predict_single_sequence(sgrna, dna, model_path='weight/final_model.keras', 
                           use_threshold=True):
    """
    Predict for a single sgRNA-DNA pair using trained Keras model.
    
    Args:
        sgrna: sgRNA sequence (string)
        dna: DNA sequence (string)
        model_path: Path to trained model
        use_threshold: Whether to use adaptive threshold
    
    Returns:
        Predicted class and probabilities
    """
    print("=" * 60)
    print("Single Sequence Prediction (Keras Model)")
    print("=" * 60)
    
    # Load model
    model = load_trained_model(model_path)
    
    # Load threshold
    threshold = load_threshold() if use_threshold else 0.5
    
    # Encode the sequences
    cnn_input = encode_for_cnn(sgrna, dna)  # (26, 7)
    token_ids = encode_for_bert(sgrna, dna)  # (26,)
    segment_ids = np.zeros(26, dtype=np.int32)
    position_ids = np.arange(26, dtype=np.int32)
    
    # Add batch dimension
    inputs = {
        'cnn_input': cnn_input[np.newaxis, ...],
        'token_ids': token_ids[np.newaxis, ...],
        'segment_ids': segment_ids[np.newaxis, ...],
        'position_ids': position_ids[np.newaxis, ...]
    }
    
    # Make prediction
    probabilities = model.predict(inputs, verbose=0)
    
    # Apply threshold
    if use_threshold:
        predicted_class = (probabilities[0, 1] >= threshold).astype(int)
    else:
        predicted_class = np.argmax(probabilities[0])
    
    print(f"sgRNA: {sgrna}")
    print(f"DNA:   {dna}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: Class 0 = {probabilities[0][0]:.4f}, Class 1 = {probabilities[0][1]:.4f}")
    
    return predicted_class, probabilities[0]


# ========== OPTION 2: Batch Prediction ==========

def predict_batch(sgrna_list, dna_list, model_path='weight/final_model.keras',
                 use_threshold=True):
    """
    Predict for multiple sgRNA-DNA pairs using trained Keras model.
    
    Args:
        sgrna_list: List of sgRNA sequences
        dna_list: List of DNA sequences
        model_path: Path to trained model
        use_threshold: Whether to use adaptive threshold
    
    Returns:
        Predicted classes and probabilities
    """
    print("\n" + "=" * 60)
    print("Batch Prediction (Keras Model)")
    print("=" * 60)
    
    # Load model
    model = load_trained_model(model_path)
    
    # Load threshold
    threshold = load_threshold() if use_threshold else 0.5
    
    batch_size = len(sgrna_list)
    
    # Encode all sequences
    cnn_inputs = np.array([encode_for_cnn(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    token_ids = np.array([encode_for_bert(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    segment_ids = np.zeros((batch_size, 26), dtype=np.int32)
    position_ids = np.tile(np.arange(26), (batch_size, 1))
    
    print(f"Batch size: {batch_size}")
    print(f"CNN input shape: {cnn_inputs.shape}")
    print(f"Token IDs shape: {token_ids.shape}")
    
    # Make predictions
    inputs = {
        'cnn_input': cnn_inputs,
        'token_ids': token_ids,
        'segment_ids': segment_ids,
        'position_ids': position_ids
    }
    
    probabilities = model.predict(inputs, verbose=0)
    
    # Apply threshold
    if use_threshold:
        predicted_classes = (probabilities[:, 1] >= threshold).astype(int)
    else:
        predicted_classes = np.argmax(probabilities, axis=1)
    
    print("\nResults:")
    for i in range(min(batch_size, 10)):  # Show first 10
        print(f"Sample {i+1}: Class {predicted_classes[i]}, Prob = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")
    
    if batch_size > 10:
        print(f"... and {batch_size - 10} more samples")
    
    return predicted_classes, probabilities


# ========== OPTION 3: Load from Dataset File ==========

def predict_from_dataset(file_path, max_samples=None, model_path='weight/final_model.keras',
                        use_threshold=True):
    """
    Load data from txt file and make predictions using trained Keras model.
    
    Args:
        file_path: Path to dataset file (e.g., 'datasets/I1.txt')
        max_samples: Maximum number of samples to process (None = all)
        model_path: Path to trained model
        use_threshold: Whether to use adaptive threshold
    
    Returns:
        Predictions, probabilities, and true labels
    """
    print("\n" + "=" * 60)
    print("Dataset Prediction (Keras Model)")
    print("=" * 60)
    
    # Load model
    model = load_trained_model(model_path)
    
    # Load threshold
    threshold = load_threshold() if use_threshold else 0.5
    
    # Load dataset
    sgrna_list, dna_list, true_labels = load_dataset(file_path, max_samples)
    print(f"Loaded {len(sgrna_list)} samples from {file_path}")
    
    # Encode sequences
    cnn_inputs = np.array([encode_for_cnn(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    token_ids = np.array([encode_for_bert(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    segment_ids = np.zeros((len(sgrna_list), 26), dtype=np.int32)
    position_ids = np.tile(np.arange(26), (len(sgrna_list), 1))
    
    # Make predictions
    inputs = {
        'cnn_input': cnn_inputs,
        'token_ids': token_ids,
        'segment_ids': segment_ids,
        'position_ids': position_ids
    }
    
    probabilities = model.predict(inputs, verbose=0)
    
    # Apply threshold
    if use_threshold:
        predicted_classes = (probabilities[:, 1] >= threshold).astype(int)
    else:
        predicted_classes = np.argmax(probabilities, axis=1)
    
    print("\nFirst 5 predictions:")
    for i in range(min(5, len(predicted_classes))):
        print(f"Sample {i+1}: Predicted = {predicted_classes[i]}, True = {int(true_labels[i])}, Prob = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")
    
    # Calculate metrics (if labels are available)
    if len(true_labels) > 0:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
        
        accuracy = accuracy_score(true_labels.astype(int), predicted_classes)
        try:
            auroc = roc_auc_score(true_labels.astype(int), probabilities[:, 1])
        except:
            auroc = 0.0
        f1 = f1_score(true_labels.astype(int), predicted_classes, average='binary')
        mcc = matthews_corrcoef(true_labels.astype(int), predicted_classes)
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUROC:    {auroc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  MCC:      {mcc:.4f}")
    
    return predicted_classes, probabilities, true_labels


# ========== MAIN ==========

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRISPR-BERT Model Inference (Using Keras Weights)")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('weight/final_model.keras'):
        print("\n⚠️  WARNING: No trained model found!")
        print("Please run train_model.py first to train the model.")
        print("\nYou can train the model by running:")
        print("  python train_model.py")
        exit(1)
    
    # Example 1: Single prediction
    sgrna = "GGTGAGTGAGTGTGTGCGTGTGG"
    dna = "TGTGAGTGTGTGTGTGTGTGTGT"
    predict_single_sequence(sgrna, dna)
    
    # Example 2: Batch prediction
    sgrna_list = [
        "GGTGAGTGAGTGTGTGCGTGTGG",
        "GCCTCTTTCCCACCCACCTTGGG",
        "GACTTGTTTTCATTGTTCTCAGG"
    ]
    dna_list = [
        "TGTGAGTGTGTGTGTGTGTGTGT",
        "GTCTCTTTCCCAGCGACCTGGGG",
        "GAGTCATTTTCATTGTCTTCATG"
    ]
    predict_batch(sgrna_list, dna_list)
    
    # Example 3: Load from dataset
    if os.path.exists('datasets/sam.txt'):
        predict_from_dataset('datasets/sam.txt', max_samples=100)
    else:
        print("\nSkipping dataset prediction (datasets/sam.txt not found)")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
