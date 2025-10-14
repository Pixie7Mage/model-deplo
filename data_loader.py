"""
Data Loader for CRISPR Datasets
Loads sequences from txt files and encodes them for CNN/BERT models
"""

import numpy as np
from sequence_encoder import encode_batch_for_cnn, encode_batch_for_bert


def load_dataset(file_path, max_samples=None):
    """
    Load sgRNA-DNA pairs from dataset file.
    
    Args:
        file_path (str): Path to dataset file (e.g., 'datasets/I1.txt')
        max_samples (int): Maximum number of samples to load (None = all)
    
    Returns:
        tuple: (sgrna_list, dna_list, labels)
    """
    sgrna_list = []
    dna_list = []
    labels = []
    
    # Read file line by line
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            
            # Parse line: sgRNA,DNA,label or sgRNA,DNA,label,additional_cols
            parts = line.split(',')
            if len(parts) >= 3:
                sgrna = parts[0]
                dna = parts[1]
                label = float(parts[2])
                
                sgrna_list.append(sgrna)
                dna_list.append(dna)
                labels.append(label)
    
    return sgrna_list, dna_list, np.array(labels)


def load_and_encode_for_cnn(file_path, max_samples=None):
    """
    Load dataset and encode for CNN model.
    All sequences encoded to fixed size (26, 7).
    
    Args:
        file_path (str): Path to dataset file
        max_samples (int): Maximum samples to load
    
    Returns:
        tuple: (X_cnn, y) where X_cnn is shape (n_samples, 26, 7)
    """
    # Step 1: Load raw sequences
    print(f"Loading dataset from {file_path}...")
    sgrna_list, dna_list, labels = load_dataset(file_path, max_samples)
    print(f"  Loaded {len(sgrna_list)} sequences")
    
    # Step 2: Encode for CNN (fixed size: 26x7)
    print(f"Encoding for CNN...")
    X_cnn = encode_batch_for_cnn(sgrna_list, dna_list)
    print(f"  CNN encoded shape: {X_cnn.shape}")
    
    return X_cnn, labels


def load_and_encode_for_bert(file_path, max_samples=None):
    """
    Load dataset and encode for BERT model.
    All sequences encoded to fixed size (26,).
    
    Args:
        file_path (str): Path to dataset file
        max_samples (int): Maximum samples to load
    
    Returns:
        tuple: (X_bert, y) where X_bert is shape (n_samples, 26)
    """
    # Step 1: Load raw sequences
    print(f"Loading dataset from {file_path}...")
    sgrna_list, dna_list, labels = load_dataset(file_path, max_samples)
    print(f"  Loaded {len(sgrna_list)} sequences")
    
    # Step 2: Encode for BERT (fixed size: 26)
    print(f"Encoding for BERT...")
    X_bert = encode_batch_for_bert(sgrna_list, dna_list)
    print(f"  BERT encoded shape: {X_bert.shape}")
    
    return X_bert, labels
