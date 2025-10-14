"""
Example: How to run CRISPR-BERT model
"""

import numpy as np
from sequence_encoder import encode_for_cnn, encode_for_bert
from crispr_bert import CRISPR_BERT
from data_loader import load_dataset

# ========== OPTION 1: Single Prediction ==========

def predict_single_sequence(sgrna, dna):
    """
    Predict for a single sgRNA-DNA pair.
    
    Args:
        sgrna: sgRNA sequence (string)
        dna: DNA sequence (string)
    
    Returns:
        Predicted class and probabilities
    """
    print("=" * 60)
    print("Single Sequence Prediction")
    print("=" * 60)
    
    # Step 1: Encode the sequences
    cnn_input = encode_for_cnn(sgrna, dna)  # (26, 7)
    token_ids = encode_for_bert(sgrna, dna)  # (26,)
    segment_ids = np.zeros(26, dtype=np.int32)  # All zeros
    position_ids = np.arange(26, dtype=np.int32)  # 0 to 25
    
    # Step 2: Add batch dimension
    cnn_input = cnn_input[np.newaxis, ...]  # (1, 26, 7)
    token_ids = token_ids[np.newaxis, ...]  # (1, 26)
    segment_ids = segment_ids[np.newaxis, ...]  # (1, 26)
    position_ids = position_ids[np.newaxis, ...]  # (1, 26)
    
    # Step 3: Initialize model
    model = CRISPR_BERT(
        vocab_size=28,
        bert_embed_dim=256,
        bert_num_heads=4,
        bert_num_layers=2,
        bert_ff_dim=1024
    )
    
    # Step 4: Make prediction
    probabilities = model.predict(cnn_input, token_ids, segment_ids, position_ids)
    predicted_class = model.predict_class(cnn_input, token_ids, segment_ids, position_ids)
    
    print(f"sgRNA: {sgrna}")
    print(f"DNA:   {dna}")
    print(f"Predicted class: {predicted_class[0]}")
    print(f"Probabilities: Class 0 = {probabilities[0][0]:.4f}, Class 1 = {probabilities[0][1]:.4f}")
    
    return predicted_class[0], probabilities[0]


# ========== OPTION 2: Batch Prediction ==========

def predict_batch(sgrna_list, dna_list):
    """
    Predict for multiple sgRNA-DNA pairs.
    
    Args:
        sgrna_list: List of sgRNA sequences
        dna_list: List of DNA sequences
    
    Returns:
        Predicted classes and probabilities
    """
    print("\n" + "=" * 60)
    print("Batch Prediction")
    print("=" * 60)
    
    batch_size = len(sgrna_list)
    
    # Step 1: Encode all sequences
    cnn_inputs = np.array([encode_for_cnn(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    token_ids = np.array([encode_for_bert(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    segment_ids = np.zeros((batch_size, 26), dtype=np.int32)
    position_ids = np.tile(np.arange(26), (batch_size, 1))
    
    print(f"Batch size: {batch_size}")
    print(f"CNN input shape: {cnn_inputs.shape}")
    print(f"Token IDs shape: {token_ids.shape}")
    
    # Step 2: Initialize model
    model = CRISPR_BERT(
        vocab_size=28,
        bert_embed_dim=256,
        bert_num_heads=4,
        bert_num_layers=2,
        bert_ff_dim=1024
    )
    
    # Step 3: Make predictions
    probabilities = model.predict(cnn_inputs, token_ids, segment_ids, position_ids)
    predicted_classes = model.predict_class(cnn_inputs, token_ids, segment_ids, position_ids)
    
    print("\nResults:")
    for i in range(batch_size):
        print(f"Sample {i+1}: Class {predicted_classes[i]}, Prob = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")
    
    return predicted_classes, probabilities


# ========== OPTION 3: Load from Dataset File ==========

def predict_from_dataset(file_path, max_samples=10):
    """
    Load data from txt file and make predictions.
    
    Args:
        file_path: Path to dataset file (e.g., 'datasets/I1.txt')
        max_samples: Maximum number of samples to process
    
    Returns:
        Predictions and true labels
    """
    print("\n" + "=" * 60)
    print("Dataset Prediction")
    print("=" * 60)
    
    # Step 1: Load dataset
    sgrna_list, dna_list, true_labels = load_dataset(file_path, max_samples)
    print(f"Loaded {len(sgrna_list)} samples from {file_path}")
    
    # Step 2: Encode sequences
    cnn_inputs = np.array([encode_for_cnn(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    token_ids = np.array([encode_for_bert(sg, dn) for sg, dn in zip(sgrna_list, dna_list)])
    segment_ids = np.zeros((len(sgrna_list), 26), dtype=np.int32)
    position_ids = np.tile(np.arange(26), (len(sgrna_list), 1))
    
    # Step 3: Initialize model
    model = CRISPR_BERT(
        vocab_size=28,
        bert_embed_dim=256,
        bert_num_heads=4,
        bert_num_layers=2,
        bert_ff_dim=1024
    )
    
    # Step 4: Make predictions
    probabilities = model.predict(cnn_inputs, token_ids, segment_ids, position_ids)
    predicted_classes = model.predict_class(cnn_inputs, token_ids, segment_ids, position_ids)
    
    print("\nFirst 5 predictions:")
    for i in range(min(5, len(predicted_classes))):
        print(f"Sample {i+1}: Predicted = {predicted_classes[i]}, True = {int(true_labels[i])}, Prob = [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}]")
    
    # Calculate accuracy (if labels are available)
    if len(true_labels) > 0:
        accuracy = np.mean(predicted_classes == true_labels)
        print(f"\nAccuracy: {accuracy:.2%}")
    
    return predicted_classes, probabilities, true_labels


# ========== MAIN ==========

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRISPR-BERT Model Usage Examples")
    print("=" * 60)
    
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
    predict_from_dataset('datasets/I1.txt', max_samples=10)
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
