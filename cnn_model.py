"""
CNN Model for CRISPR Off-Target Prediction
Built from scratch using the architecture from Figure S4
"""

import numpy as np


class Conv2D:
    """
    2D Convolution layer from scratch.
    Applies sliding window convolution over input.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output filters
            kernel_size: Size of convolution kernel (height, width)
            stride: Stride for convolution
            padding: Padding size
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle single int or tuple for kernel_size
        if isinstance(kernel_size, int):
            self.kernel_h, self.kernel_w = kernel_size, kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        self.stride = stride
        self.padding = padding
        
        # Initialize weights: Xavier initialization
        # Shape: (out_channels, in_channels, kernel_h, kernel_w)
        limit = np.sqrt(6.0 / (in_channels * self.kernel_h * self.kernel_w + out_channels))
        self.weights = np.random.uniform(-limit, limit, 
                                        (out_channels, in_channels, self.kernel_h, self.kernel_w))
        self.bias = np.zeros(out_channels)
        
        # For backprop
        self.input = None
        self.output = None
    
    def forward(self, x):
        """
        Forward pass: convolve input with filters.
        
        Args:
            x: Input tensor of shape (batch, height, width, in_channels)
        
        Returns:
            Output tensor of shape (batch, out_h, out_w, out_channels)
        """
        batch_size, h, w, _ = x.shape
        self.input = x
        
        # Add padding if needed
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding), (0, 0)), mode='constant')
        
        # Calculate output dimensions
        out_h = (h + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_w) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, self.out_channels))
        
        # Perform convolution
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    # Extract window
                    h_start = i * self.stride
                    w_start = j * self.stride
                    window = x[b, h_start:h_start+self.kernel_h, 
                              w_start:w_start+self.kernel_w, :]
                    
                    # Convolve with each filter
                    for f in range(self.out_channels):
                        # Element-wise multiply and sum
                        output[b, i, j, f] = np.sum(window * self.weights[f]) + self.bias[f]
        
        self.output = output
        return output


class ReLU:
    """ReLU activation function."""
    
    def forward(self, x):
        """Apply ReLU: max(0, x)"""
        return np.maximum(0, x)


class MaxPool2D:
    """Max pooling layer."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        """
        Apply max pooling.
        
        Args:
            x: Input of shape (batch, h, w, channels)
        
        Returns:
            Pooled output
        """
        batch_size, h, w, channels = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    window = x[b, h_start:h_start+self.pool_size,
                              w_start:w_start+self.pool_size, :]
                    output[b, i, j, :] = np.max(window, axis=(0, 1))
        
        return output


class BiGRU:
    """
    Bidirectional GRU layer from scratch.
    
    GRU Cell Formulas (for each timestep t):
        z_t = σ(W_z @ x_t + U_z @ h_{t-1} + b_z)           # Update gate
        r_t = σ(W_r @ x_t + U_r @ h_{t-1} + b_r)           # Reset gate
        h̃_t = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)  # Candidate state
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t             # New hidden state
    
    Where:
        σ = sigmoid function
        ⊙ = element-wise multiplication
        @ = matrix multiplication
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize BiGRU layer.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Number of hidden units in each direction
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + hidden_size))
        
        # Forward GRU weights
        # Update gate
        self.W_z_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_z_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z_fwd = np.zeros(hidden_size)
        
        # Reset gate
        self.W_r_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_r_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r_fwd = np.zeros(hidden_size)
        
        # Candidate state
        self.W_h_fwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_h_fwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h_fwd = np.zeros(hidden_size)
        
        # Backward GRU weights
        # Update gate
        self.W_z_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_z_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z_bwd = np.zeros(hidden_size)
        
        # Reset gate
        self.W_r_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_r_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r_bwd = np.zeros(hidden_size)
        
        # Candidate state
        self.W_h_bwd = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.U_h_bwd = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h_bwd = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Hyperbolic tangent activation"""
        return np.tanh(np.clip(x, -500, 500))
    
    def gru_step(self, x_t, h_prev, W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h):
        """
        Single GRU step computation.
        
        Args:
            x_t: Input at timestep t, shape (input_size,)
            h_prev: Previous hidden state, shape (hidden_size,)
            W_z, U_z, b_z: Update gate parameters
            W_r, U_r, b_r: Reset gate parameters
            W_h, U_h, b_h: Candidate state parameters
        
        Returns:
            h_t: New hidden state, shape (hidden_size,)
        """
        # Update gate: z_t = σ(W_z @ x_t + U_z @ h_{t-1} + b_z)
        z_t = self.sigmoid(W_z @ x_t + U_z @ h_prev + b_z)
        
        # Reset gate: r_t = σ(W_r @ x_t + U_r @ h_{t-1} + b_r)
        r_t = self.sigmoid(W_r @ x_t + U_r @ h_prev + b_r)
        
        # Candidate hidden state: h̃_t = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)
        h_tilde = self.tanh(W_h @ x_t + U_h @ (r_t * h_prev) + b_h)
        
        # New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def forward(self, x):
        """
        Forward pass through BiGRU.
        
        Process:
            1. Forward pass: process sequence from t=0 to t=T-1
            2. Backward pass: process sequence from t=T-1 to t=0
            3. Concatenate forward and backward hidden states
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size)
        
        Returns:
            output: Hidden states of shape (batch_size, seq_len, 2*hidden_size)
                    First hidden_size dimensions are forward states
                    Last hidden_size dimensions are backward states
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize output arrays
        h_forward = np.zeros((batch_size, seq_len, self.hidden_size))
        h_backward = np.zeros((batch_size, seq_len, self.hidden_size))
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Forward direction: t = 0, 1, 2, ..., T-1
            h_prev_fwd = np.zeros(self.hidden_size)
            for t in range(seq_len):
                h_prev_fwd = self.gru_step(
                    x[b, t], h_prev_fwd,
                    self.W_z_fwd, self.U_z_fwd, self.b_z_fwd,
                    self.W_r_fwd, self.U_r_fwd, self.b_r_fwd,
                    self.W_h_fwd, self.U_h_fwd, self.b_h_fwd
                )
                h_forward[b, t] = h_prev_fwd
            
            # Backward direction: t = T-1, T-2, ..., 0
            h_prev_bwd = np.zeros(self.hidden_size)
            for t in range(seq_len - 1, -1, -1):
                h_prev_bwd = self.gru_step(
                    x[b, t], h_prev_bwd,
                    self.W_z_bwd, self.U_z_bwd, self.b_z_bwd,
                    self.W_r_bwd, self.U_r_bwd, self.b_r_bwd,
                    self.W_h_bwd, self.U_h_bwd, self.b_h_bwd
                )
                h_backward[b, t] = h_prev_bwd
        
        # Concatenate forward and backward hidden states
        # Shape: (batch_size, seq_len, 2*hidden_size)
        output = np.concatenate([h_forward, h_backward], axis=-1)
        
        return output


class Dense:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size, activation=None):
        """
        Initialize dense layer.
        
        Args:
            input_size: Size of input features
            output_size: Size of output features
            activation: Activation function ('relu', 'sigmoid', 'softmax', None)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, input_size) or (batch, seq, input_size)
        
        Returns:
            Output after linear transformation and activation
        """
        # Linear transformation: y = xW + b
        output = x @ self.weights + self.bias
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))
        elif self.activation == 'softmax':
            exp_x = np.exp(output - np.max(output, axis=-1, keepdims=True))
            output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        return output


class Dropout:
    """Dropout layer for regularization."""
    
    def __init__(self, rate=0.35):
        self.rate = rate
    
    def forward(self, x, training=True):
        """Apply dropout during training."""
        if training and self.rate > 0:
            mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * mask
        return x


class InceptionCNN:
    """
    Inception-based CNN Branch for CRISPR-BERT.
    Architecture: Input (26, 7) → Multi-scale Conv2D → Output (26, 80)
    
    Channels: 5, 15, 25, 35 (total = 80)
    Kernel sizes: 1×1, 2×2, 3×3, 5×5
    """
    
    def __init__(self):
        # Multi-scale Conv2D layers with specific channel counts
        self.conv_1x1 = Conv2D(in_channels=1, out_channels=5, kernel_size=(1, 1))
        self.conv_2x2 = Conv2D(in_channels=1, out_channels=15, kernel_size=(2, 2), padding=0)
        self.conv_3x3 = Conv2D(in_channels=1, out_channels=25, kernel_size=(3, 3), padding=1)
        self.conv_5x5 = Conv2D(in_channels=1, out_channels=35, kernel_size=(5, 5), padding=2)
        
        self.relu = ReLU()
    
    def forward(self, x):
        """
        Forward pass through Inception CNN.
        
        Args:
            x: Input of shape (batch, 26, 7) - CNN encoded sequences
        
        Returns:
            Output of shape (batch, 26, 80)
        """
        batch_size = x.shape[0]
        
        # Reshape for Conv2D: (batch, 26, 7, 1) - add channel dimension
        x = x.reshape(batch_size, 26, 7, 1)
        
        # Apply multi-scale convolutions with ReLU
        conv1_out = self.relu.forward(self.conv_1x1.forward(x))  # (batch, 26, 7, 5)
        conv2_out = self.relu.forward(self.conv_2x2.forward(x))  # (batch, 25, 6, 15)
        conv3_out = self.relu.forward(self.conv_3x3.forward(x))  # (batch, 26, 7, 25)
        conv5_out = self.relu.forward(self.conv_5x5.forward(x))  # (batch, 26, 7, 35)
        
        # Resize conv2 output to match 26×7 (pad if needed)
        if conv2_out.shape[1] != 26 or conv2_out.shape[2] != 7:
            new_conv2 = np.zeros((batch_size, 26, 7, 15))
            h, w = min(conv2_out.shape[1], 26), min(conv2_out.shape[2], 7)
            new_conv2[:, :h, :w, :] = conv2_out[:, :h, :w, :]
            conv2_out = new_conv2
        
        # Concatenate along channel dimension: 5 + 15 + 25 + 35 = 80 channels
        conv_concat = np.concatenate([conv1_out, conv2_out, conv3_out, conv5_out], axis=-1)
        # Shape: (batch, 26, 7, 80)
        
        # Average over width dimension to get (batch, 26, 80)
        output = np.mean(conv_concat, axis=2)  # (batch, 26, 80)
        
        return output
