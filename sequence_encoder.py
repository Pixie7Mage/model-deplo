"""
Sequence Encoder for CRISPR sgRNA-DNA Pairs
Converts sgRNA and DNA sequences into paired encodings for CNN and BERT models
"""

import numpy as np


# ========== ENCODING DICTIONARIES ==========

# CNN: 7-dimensional one-hot encoding for each paired token
CNN_ENCODING = {
    'AA': [1, 0, 0, 0, 0, 0, 0], 'AT': [1, 1, 0, 0, 0, 1, 0], 'AG': [1, 0, 1, 0, 0, 1, 0], 'AC': [1, 0, 0, 1, 0, 1, 0],
    'TA': [1, 1, 0, 0, 0, 0, 1], 'TT': [0, 1, 0, 0, 0, 0, 0], 'TG': [0, 1, 1, 0, 0, 1, 0], 'TC': [0, 1, 0, 1, 0, 1, 0],
    'GA': [1, 0, 1, 0, 0, 0, 1], 'GT': [0, 1, 1, 0, 0, 0, 1], 'GG': [0, 0, 1, 0, 0, 0, 0], 'GC': [0, 0, 1, 1, 0, 1, 0],
    'CA': [1, 0, 0, 1, 0, 0, 1], 'CT': [0, 1, 0, 1, 0, 0, 1], 'CG': [0, 0, 1, 1, 0, 0, 1], 'CC': [0, 0, 0, 1, 0, 0, 0],
    'A_': [1, 0, 0, 0, 1, 1, 0], 'T_': [0, 1, 0, 0, 1, 1, 0], 'G_': [0, 0, 1, 0, 1, 1, 0], 'C_': [0, 0, 0, 1, 1, 1, 0],
    '_A': [1, 0, 0, 0, 1, 0, 1], '_T': [0, 1, 0, 0, 1, 0, 1], '_G': [0, 0, 1, 0, 1, 0, 1], '_C': [0, 0, 0, 1, 1, 0, 1],
    '--': [0, 0, 0, 0, 0, 0, 0],
    '[CLS]': [0, 0, 0, 0, 0, 0, 0],  # Special token for start
    '[SEP]': [0, 0, 0, 0, 0, 0, 0],  # Special token for end
    '[PAD]': [0, 0, 0, 0, 0, 0, 0]   # Padding token
}

# Fixed sequence length (excluding special tokens)
FIXED_SEQ_LENGTH = 24  # 24 base pairs
TOTAL_LENGTH = 26      # 24 pairs + [CLS] + [SEP]

# BERT: Token ID mapping for each paired token
BERT_TOKEN_DICT = {
    "AA": 2, "AC": 3, "AG": 4, "AT": 5,
    "CA": 6, "CC": 7, "CG": 8, "CT": 9,
    "GA": 10, "GC": 11, "GG": 12, "GT": 13,
    "TA": 14, "TC": 15, "TG": 16, "TT": 17,
    "A_": 18, "_A": 19, "C_": 20, "_C": 21,
    "G_": 22, "_G": 23, "T_": 24, "_T": 25,
    "[CLS]": 0, "[SEP]": 1, "[PAD]": 27
}


# ========== SEQUENCE PAIRING ==========

def pair_sequences(sgrna, dna):
    """
    Pair sgRNA and DNA sequences character by character.
    
    Args:
        sgrna (str): sgRNA sequence (e.g., "GAGTCCGAGCAG")
        dna (str): DNA sequence (e.g., "GGAGTCCGTGCA")
    
    Returns:
        list: Paired tokens (e.g., ["GG", "GA", "AG", ...])
    """
    # Convert to uppercase for consistency
    sgrna = sgrna.upper()
    dna = dna.upper()
    
    # Ensure sequences are same length
    if len(sgrna) != len(dna):
        raise ValueError(f"Sequences must be same length: sgRNA={len(sgrna)}, DNA={len(dna)}")
    
    # Pair each position: sgRNA[i] + DNA[i]
    paired_tokens = []
    for i in range(len(sgrna)):
        pair = sgrna[i] + dna[i]
        paired_tokens.append(pair)
    
    return paired_tokens


# ========== CNN ENCODING ==========

def encode_for_cnn(sgrna, dna, fixed_length=FIXED_SEQ_LENGTH):
    """
    Encode sgRNA-DNA pair for CNN model with FIXED LENGTH.
    Returns 26x7 matrix: [CLS] + 24 base pairs + [SEP]
    
    Args:
        sgrna (str): sgRNA sequence
        dna (str): DNA sequence
        fixed_length (int): Fixed sequence length (default: 24)
    
    Returns:
        numpy.ndarray: Shape (26, 7) - binary encoded matrix
    """
    # Step 1: Pair the sequences
    paired_tokens = pair_sequences(sgrna, dna)
    
    # Step 2: Pad or truncate to fixed length (24 positions)
    if len(paired_tokens) < fixed_length:
        # Pad with [PAD] tokens
        padding_needed = fixed_length - len(paired_tokens)
        paired_tokens = paired_tokens + ['[PAD]'] * padding_needed
    elif len(paired_tokens) > fixed_length:
        # Truncate to fixed length
        paired_tokens = paired_tokens[:fixed_length]
    
    # Step 3: Add [CLS] at start and [SEP] at end
    paired_tokens = ['[CLS]'] + paired_tokens + ['[SEP]']
    
    # Step 4: Convert each token to 7-dim vector
    encoded_sequence = []
    for token in paired_tokens:
        if token in CNN_ENCODING:
            encoded_sequence.append(CNN_ENCODING[token])
        else:
            # Unknown token: use all zeros
            encoded_sequence.append([0, 0, 0, 0, 0, 0, 0])
    
    # Step 5: Convert to numpy array - should be (26, 7)
    result = np.array(encoded_sequence, dtype=np.float32)
    assert result.shape == (TOTAL_LENGTH, 7), f"Expected shape (26, 7), got {result.shape}"
    return result


# ========== BERT ENCODING ==========

def encode_for_bert(sgrna, dna, fixed_length=FIXED_SEQ_LENGTH):
    """
    Encode sgRNA-DNA pair for BERT model with FIXED LENGTH.
    Returns length 26: [CLS] + 24 base pairs + [SEP]
    
    Args:
        sgrna (str): sgRNA sequence
        dna (str): DNA sequence
        fixed_length (int): Fixed sequence length (default: 24)
    
    Returns:
        numpy.ndarray: Token IDs (shape: 26)
    """
    # Step 1: Pair the sequences
    paired_tokens = pair_sequences(sgrna, dna)
    
    # Step 2: Pad or truncate to fixed length (24 positions)
    if len(paired_tokens) < fixed_length:
        # Pad with [PAD] tokens
        padding_needed = fixed_length - len(paired_tokens)
        paired_tokens = paired_tokens + ['[PAD]'] * padding_needed
    elif len(paired_tokens) > fixed_length:
        # Truncate to fixed length
        paired_tokens = paired_tokens[:fixed_length]
    
    # Step 3: Add [CLS] at start and [SEP] at end
    paired_tokens = ['[CLS]'] + paired_tokens + ['[SEP]']
    
    # Step 4: Convert each token to token ID
    token_ids = []
    for token in paired_tokens:
        if token in BERT_TOKEN_DICT:
            token_ids.append(BERT_TOKEN_DICT[token])
        else:
            # Unknown token: use [PAD] token ID
            token_ids.append(BERT_TOKEN_DICT["[PAD]"])
    
    # Step 5: Convert to numpy array - should be (26,)
    result = np.array(token_ids, dtype=np.int32)
    assert result.shape == (TOTAL_LENGTH,), f"Expected shape (26,), got {result.shape}"
    return result


# ========== BATCH ENCODING ==========

def encode_batch_for_cnn(sgrna_list, dna_list):
    """
    Encode multiple sgRNA-DNA pairs for CNN model.
    All sequences are encoded to fixed size (26, 7).
    
    Args:
        sgrna_list (list): List of sgRNA sequences
        dna_list (list): List of DNA sequences
    
    Returns:
        numpy.ndarray: Shape (batch_size, 26, 7)
    """
    # Encode all sequences - each will be (26, 7)
    batch_encoded = []
    for sgrna, dna in zip(sgrna_list, dna_list):
        encoded = encode_for_cnn(sgrna, dna)  # Returns (26, 7)
        batch_encoded.append(encoded)
    
    # Stack into batch
    return np.array(batch_encoded, dtype=np.float32)


def encode_batch_for_bert(sgrna_list, dna_list):
    """
    Encode multiple sgRNA-DNA pairs for BERT model.
    All sequences are encoded to fixed size (26,).
    
    Args:
        sgrna_list (list): List of sgRNA sequences
        dna_list (list): List of DNA sequences
    
    Returns:
        numpy.ndarray: Shape (batch_size, 26)
    """
    # Encode all sequences - each will be (26,)
    batch_encoded = []
    for sgrna, dna in zip(sgrna_list, dna_list):
        encoded = encode_for_bert(sgrna, dna)  # Returns (26,)
        batch_encoded.append(encoded)
    
    # Stack into batch
    return np.array(batch_encoded, dtype=np.int32)


# ========== UTILITY FUNCTIONS ==========

def decode_cnn_encoding(encoded_matrix):
    """
    Decode CNN encoded matrix back to paired tokens (for debugging).
    
    Args:
        encoded_matrix (numpy.ndarray): Shape (seq_length, 7)
    
    Returns:
        list: Paired tokens
    """
    # Create reverse mapping
    reverse_map = {tuple(v): k for k, v in CNN_ENCODING.items()}
    
    decoded_tokens = []
    for vector in encoded_matrix:
        key = tuple(vector.astype(int).tolist())
        token = reverse_map.get(key, "??")
        decoded_tokens.append(token)
    
    return decoded_tokens


def decode_bert_encoding(token_ids):
    """
    Decode BERT token IDs back to paired tokens (for debugging).
    
    Args:
        token_ids (numpy.ndarray): Token IDs
    
    Returns:
        list: Paired tokens
    """
    # Create reverse mapping
    reverse_map = {v: k for k, v in BERT_TOKEN_DICT.items()}
    
    decoded_tokens = []
    for token_id in token_ids:
        token = reverse_map.get(int(token_id), "??")
        decoded_tokens.append(token)
    
    return decoded_tokens
