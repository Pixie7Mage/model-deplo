"""
BERT Model for CRISPR Off-Target Prediction
Built from scratch using the architecture from Figure S5
"""

import numpy as np


class Embedding:
    """Embedding layer to convert token IDs to vectors."""
    
    def __init__(self, vocab_size, embed_dim):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embedding matrix
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def forward(self, x):
        """
        Forward pass: look up embeddings for input IDs.
        
        Args:
            x: Input token IDs of shape (batch, seq_len)
        
        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.embeddings[x]


class LayerNorm:
    """
    Layer Normalization.
    Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    
    def __init__(self, normalized_shape, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
    
    def forward(self, x):
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.
    
    Formulas:
        Q = XW_q, K = XW_k, V = XW_v
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
        MultiHead = Concat(head_1, ..., head_h) W_o
    """
    
    def __init__(self, embed_dim, num_heads):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Weight matrices for Q, K, V
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01
        
        # Output projection
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Scaled dot-product attention.
        
        Args:
            Q, K, V: Query, Key, Value matrices
        
        Returns:
            Attention output
        """
        # Calculate attention scores: QK^T / sqrt(d_k)
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
        
        # Apply softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = attention_weights @ V
        
        return output
    
    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input of shape (batch, seq_len, embed_dim)
        
        Returns:
            Output of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = x @ self.W_q  # (batch, seq_len, embed_dim)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads: (batch, seq_len, embed_dim)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = attention_output @ self.W_o
        
        return output


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    Formula: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, embed_dim, ff_dim):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Embedding dimension
            ff_dim: Hidden dimension of feed-forward network
        """
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.01
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.01
        self.b2 = np.zeros(embed_dim)
    
    def gelu(self, x):
        """
        GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        Approximation used in BERT
        """
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x):
        """Forward pass through feed-forward network."""
        hidden = self.gelu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output


class TransformerBlock:
    """
    Transformer encoder block.
    
    Architecture:
        x -> [Multi-Head Attention -> Add & Norm] -> [FFN -> Add & Norm] -> output
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm2 = LayerNorm(embed_dim)
        
        self.dropout_rate = dropout_rate
    
    def dropout(self, x, training=True):
        """Apply dropout."""
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x, training=True):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input of shape (batch, seq_len, embed_dim)
            training: Whether in training mode
        
        Returns:
            Output of shape (batch, seq_len, embed_dim)
        """
        # Multi-head attention with residual connection
        attn_output = self.attention.forward(x)
        attn_output = self.dropout(attn_output, training)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn.forward(x)
        ffn_output = self.dropout(ffn_output, training)
        x = self.norm2.forward(x + ffn_output)
        
        return x


class BiGRU:
    """
    Bidirectional GRU layer.
    Same implementation as in CNN model.
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for forward GRU
        limit = np.sqrt(1.0 / hidden_size)
        self.W_z_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_z_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z_fwd = np.zeros(hidden_size)
        
        self.W_r_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_r_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r_fwd = np.zeros(hidden_size)
        
        self.W_h_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_h_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h_fwd = np.zeros(hidden_size)
        
        # Initialize weights for backward GRU
        self.W_z_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_z_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z_bwd = np.zeros(hidden_size)
        
        self.W_r_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_r_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r_bwd = np.zeros(hidden_size)
        
        self.W_h_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_h_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h_bwd = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh activation."""
        return np.tanh(np.clip(x, -500, 500))
    
    def gru_cell_forward(self, x_t, h_prev, direction='fwd'):
        """Single GRU cell forward pass."""
        if direction == 'fwd':
            W_z, U_z, b_z = self.W_z_fwd, self.U_z_fwd, self.b_z_fwd
            W_r, U_r, b_r = self.W_r_fwd, self.U_r_fwd, self.b_r_fwd
            W_h, U_h, b_h = self.W_h_fwd, self.U_h_fwd, self.b_h_fwd
        else:
            W_z, U_z, b_z = self.W_z_bwd, self.U_z_bwd, self.b_z_bwd
            W_r, U_r, b_r = self.W_r_bwd, self.U_r_bwd, self.b_r_bwd
            W_h, U_h, b_h = self.W_h_bwd, self.U_h_bwd, self.b_h_bwd
        
        # GRU formulas
        z_t = self.sigmoid(W_z @ x_t + U_z @ h_prev + b_z)
        r_t = self.sigmoid(W_r @ x_t + U_r @ h_prev + b_r)
        h_tilde = self.tanh(W_h @ x_t + U_h @ (r_t * h_prev) + b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def forward(self, x):
        """Forward pass through BiGRU."""
        batch_size, seq_len, _ = x.shape
        
        # Initialize outputs
        h_fwd = np.zeros((batch_size, seq_len, self.hidden_size))
        h_bwd = np.zeros((batch_size, seq_len, self.hidden_size))
        
        for b in range(batch_size):
            # Forward direction
            h_prev = np.zeros(self.hidden_size)
            for t in range(seq_len):
                h_prev = self.gru_cell_forward(x[b, t], h_prev, 'fwd')
                h_fwd[b, t] = h_prev
            
            # Backward direction
            h_prev = np.zeros(self.hidden_size)
            for t in range(seq_len - 1, -1, -1):
                h_prev = self.gru_cell_forward(x[b, t], h_prev, 'bwd')
                h_bwd[b, t] = h_prev
        
        # Concatenate forward and backward
        output = np.concatenate([h_fwd, h_bwd], axis=-1)
        return output


class Dense:
    """Fully connected layer."""
    
    def __init__(self, input_size, output_size, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        """Forward pass."""
        output = x @ self.weights + self.bias
        
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))
        elif self.activation == 'softmax':
            exp_x = np.exp(output - np.max(output, axis=-1, keepdims=True))
            output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        return output


class Dropout:
    """Dropout layer."""
    
    def __init__(self, rate=0.35):
        self.rate = rate
    
    def forward(self, x, training=True):
        """Apply dropout."""
        if training and self.rate > 0:
            mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * mask
        return x


class BERTBranch:
    """
    BERT Branch for CRISPR-BERT.
    Architecture: Token/Position/Segment Embeddings → Transformer → Output (26, 80)
    
    Note: Original uses BERT-base (768 hidden), but we project to 80 dimensions
    """
    
    def __init__(self, vocab_size=28, embed_dim=256, num_heads=4, num_layers=2, ff_dim=1024, output_dim=80):
        """
        Initialize BERT branch.
        
        Args:
            vocab_size: Size of token vocabulary (default: 28 for CRISPR tokens)
            embed_dim: Embedding dimension (default: 256)
            num_heads: Number of attention heads (default: 4)
            num_layers: Number of transformer layers (default: 2)
            ff_dim: Feed-forward dimension (default: 1024)
            output_dim: Output dimension (default: 80)
        """
        # Embedding layers
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Embedding(26, embed_dim)  # Max sequence length = 26
        self.segment_embedding = Embedding(2, embed_dim)  # Segment A/B (we use only A)
        
        self.embed_norm = LayerNorm(embed_dim)
        self.embed_dropout = Dropout(0.1)
        
        # Transformer layers
        self.transformer_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate=0.1)
            for _ in range(num_layers)
        ]
        
        # Projection layer to output dimension (embed_dim → 80)
        self.projection = Dense(embed_dim, output_dim, activation=None)
    
    def forward(self, token_ids, segment_ids, position_ids, training=True):
        """
        Forward pass through BERT branch.
        
        Args:
            token_ids: Token IDs of shape (batch, 26)
            segment_ids: Segment IDs of shape (batch, 26) - all zeros
            position_ids: Position IDs of shape (batch, 26) - [0, 1, ..., 25]
            training: Whether in training mode
        
        Returns:
            Output of shape (batch, 26, 80)
        """
        batch_size = token_ids.shape[0]
        
        # Embedding layer: sum of token, position, and segment embeddings
        token_embeds = self.token_embedding.forward(token_ids)
        position_embeds = self.position_embedding.forward(position_ids)
        segment_embeds = self.segment_embedding.forward(segment_ids)
        
        # Sum embeddings
        embeddings = token_embeds + position_embeds + segment_embeds
        
        # Apply layer norm and dropout
        embeddings = self.embed_norm.forward(embeddings)
        embeddings = self.embed_dropout.forward(embeddings, training)
        
        # Pass through transformer layers
        x = embeddings
        for layer in self.transformer_layers:
            x = layer.forward(x, training)
        
        # Project to output dimension: (batch, 26, embed_dim) → (batch, 26, 80)
        output = self.projection.forward(x)
        
        return output
