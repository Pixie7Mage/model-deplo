"""
CRISPR-BERT: Combined Model for CRISPR Off-Target Prediction
Combines Inception-based CNN and BERT branches
"""

import numpy as np
from cnn_model import InceptionCNN, BiGRU, Dense, Dropout
from bert_model import BERTBranch


class CRISPR_BERT:
    """
    CRISPR-BERT Model.
    
    Architecture (from paper):
        (i) Input: sgRNA-DNA pairs
        (ii) Encoding: CNN (26×7) and BERT (token IDs + segment IDs + position IDs)
        (iii) CNN Branch → (26×80) and BERT Branch → (26×80)
        (iv) Separate BiGRU layers:
             - CNN branch → BiGRU (20 forward + 20 backward) → (26×40)
             - BERT branch → BiGRU (20 forward + 20 backward) → (26×40)
        (v) Take last timestep and apply weights:
             - CNN BiGRU output × 0.2 → (40,)
             - BERT BiGRU output × 0.8 → (40,)
             - Concatenate → (80,)
        (vi) Dense layers (80→128→64→2) with dropout (0.35)
        (vii) Output: Binary classification
    """
    
    def __init__(self, 
                 vocab_size=28, 
                 bert_embed_dim=256, 
                 bert_num_heads=4, 
                 bert_num_layers=2,
                 bert_ff_dim=1024):
        """
        Initialize CRISPR-BERT model.
        
        Args:
            vocab_size: Size of token vocabulary
            bert_embed_dim: BERT embedding dimension
            bert_num_heads: Number of attention heads in BERT
            bert_num_layers: Number of transformer layers
            bert_ff_dim: Feed-forward dimension in BERT
        """
        # CNN Branch: (26, 7) → (26, 80)
        self.cnn_branch = InceptionCNN()
        
        # BERT Branch: token_ids → (26, 80)
        self.bert_branch = BERTBranch(
            vocab_size=vocab_size,
            embed_dim=bert_embed_dim,
            num_heads=bert_num_heads,
            num_layers=bert_num_layers,
            ff_dim=bert_ff_dim,
            output_dim=80
        )
        
        # Separate BiGRU layers for each branch
        # Each BiGRU: 20 forward + 20 backward = 40 output dimensions
        self.cnn_bigru = BiGRU(input_size=80, hidden_size=20)
        self.bert_bigru = BiGRU(input_size=80, hidden_size=20)
        
        # Weights for combining BiGRU outputs
        self.w1 = 0.2  # Weight for CNN branch
        self.w2 = 0.8  # Weight for BERT branch
        
        # Dense layers with dropout
        # Input: 40 (CNN BiGRU weighted) + 40 (BERT BiGRU weighted) = 80
        self.dropout1 = Dropout(rate=0.35)
        self.dense1 = Dense(80, 128, activation='relu')
        
        self.dropout2 = Dropout(rate=0.35)
        self.dense2 = Dense(128, 64, activation='relu')
        
        self.dense3 = Dense(64, 2, activation='softmax')
    
    def forward(self, cnn_input, token_ids, segment_ids, position_ids, training=True):
        """
        Forward pass through CRISPR-BERT.
        
        Args:
            cnn_input: CNN encoded input of shape (batch, 26, 7)
            token_ids: Token IDs of shape (batch, 26)
            segment_ids: Segment IDs of shape (batch, 26)
            position_ids: Position IDs of shape (batch, 26)
            training: Whether in training mode
        
        Returns:
            Output probabilities of shape (batch, 2)
        """
        # Step 1: CNN Branch
        cnn_features = self.cnn_branch.forward(cnn_input)  # (batch, 26, 80)
        
        # Step 2: BERT Branch
        bert_features = self.bert_branch.forward(
            token_ids, segment_ids, position_ids, training
        )  # (batch, 26, 80)
        
        # Step 3: Separate BiGRU for each branch
        # CNN BiGRU: (batch, 26, 80) → (batch, 26, 40)
        # 20 forward + 20 backward = 40
        cnn_gru_out = self.cnn_bigru.forward(cnn_features)  # (batch, 26, 40)
        
        # BERT BiGRU: (batch, 26, 80) → (batch, 26, 40)
        # 20 forward + 20 backward = 40
        bert_gru_out = self.bert_bigru.forward(bert_features)  # (batch, 26, 40)
        
        # Step 4: Take last timestep from each BiGRU
        cnn_last = cnn_gru_out[:, -1, :]   # (batch, 40)
        bert_last = bert_gru_out[:, -1, :]  # (batch, 40)
        
        # Step 5: Apply weights and concatenate
        # w1 = 0.2 for CNN, w2 = 0.8 for BERT
        cnn_weighted = self.w1 * cnn_last    # (batch, 40)
        bert_weighted = self.w2 * bert_last  # (batch, 40)
        
        combined = np.concatenate([cnn_weighted, bert_weighted], axis=-1)  # (batch, 80)
        
        # Step 6: Dense layers with dropout
        x = self.dropout1.forward(combined, training)
        x = self.dense1.forward(x)  # (batch, 128)
        
        x = self.dropout2.forward(x, training)
        x = self.dense2.forward(x)  # (batch, 64)
        
        output = self.dense3.forward(x)  # (batch, 2)
        
        return output
    
    def predict(self, cnn_input, token_ids, segment_ids, position_ids):
        """
        Make predictions (inference mode).
        
        Args:
            cnn_input: CNN encoded input of shape (batch, 26, 7)
            token_ids: Token IDs of shape (batch, 26)
            segment_ids: Segment IDs of shape (batch, 26)
            position_ids: Position IDs of shape (batch, 26)
        
        Returns:
            Predicted probabilities of shape (batch, 2)
        """
        return self.forward(cnn_input, token_ids, segment_ids, position_ids, training=False)
    
    def predict_class(self, cnn_input, token_ids, segment_ids, position_ids):
        """
        Predict class labels.
        
        Returns:
            Predicted class (0 or 1) for each sample
        """
        probs = self.predict(cnn_input, token_ids, segment_ids, position_ids)
        return np.argmax(probs, axis=1)
