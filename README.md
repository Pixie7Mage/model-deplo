# CRISPR-BERT: Off-Target Prediction Model

Deep learning model for predicting CRISPR off-target effects using a hybrid CNN-BERT architecture.

---

## Overview

CRISPR-BERT combines Inception-based CNN and BERT transformer architectures to predict off-target activity of CRISPR-Cas9 guide RNAs. The model processes sgRNA-DNA pairs and outputs binary classification (on-target vs off-target).

### Key Features

- **Hybrid Architecture:** CNN branch (20%) + BERT branch (80%)
- **Inception CNN:** Multi-scale convolution (1×1, 3×1, 5×1 filters)
- **BERT Transformers:** Multi-head attention with position embeddings
- **BiGRU Layers:** Bidirectional sequence processing
- **Comprehensive Metrics:** AUROC, PRAUC, F1, MCC reporting
- **Adaptive Balancing:** Dynamic class weighting and threshold optimization

---

## Project Structure

```
d:/scratch/
├── train_model.py          # Training script with metrics
├── run_model.py            # Inference script (Keras models)
├── data_loader.py          # Dataset loading utilities
├── sequence_encoder.py     # Sequence encoding (CNN & BERT)
├── crispr_bert.py          # NumPy-based model (legacy)
├── bert_model.py           # BERT transformer implementation
├── cnn_model.py            # CNN implementation
├── datasets/               # Training datasets
│   └── sam.txt            # Example dataset
└── weight/                 # Saved models
    ├── final_model.h5
    ├── best_model.h5
    └── threshold_schedule.json
```

---

## Requirements

```bash
pip install numpy tensorflow scikit-learn
```

**Recommended Versions:**
- Python 3.8+
- TensorFlow 2.x
- NumPy 1.19+
- scikit-learn 0.24+

---

## Quick Start

### 1. Train the Model

```bash
python train_model.py
```

**What it does:**
- Loads dataset from `datasets/sam.txt`
- Encodes sequences for CNN (26×7) and BERT (token IDs)
- Trains with optimized hyperparameters:
  - Learning rate: `1e-4`
  - Batch size: `256`
  - Epochs: `30` (with early stopping)
- Outputs comprehensive metrics (AUROC, PRAUC, F1, MCC)
- Saves model to `weight/final_model.h5`

**Training Output:**
```
[1/5] Loading dataset...
[2/5] Encoding sequences...
[3/5] Splitting data...
[4/5] Building model...
[5/5] Training...

Final Validation Results:
  Accuracy:    0.XXXX
  AUROC:       0.XXXX
  PRAUC:       0.XXXX
  F1 Score:    0.XXXX
  MCC:         0.XXXX
```

### 2. Run Inference

```bash
python run_model.py
```

**What it does:**
- Loads trained model from `weight/final_model.h5`
- Runs three examples:
  1. Single sequence prediction
  2. Batch prediction (3 samples)
  3. Dataset prediction (100 samples from `sam.txt`)
- Uses adaptive threshold from training

**Inference Functions:**

```python
from run_model import predict_single_sequence, predict_batch, predict_from_dataset

# Single prediction
pred_class, probs = predict_single_sequence(
    sgrna="GGTGAGTGAGTGTGTGCGTGTGG",
    dna="TGTGAGTGTGTGTGTGTGTGTGT"
)

# Batch prediction
classes, probs = predict_batch(sgrna_list, dna_list)

# Dataset evaluation with metrics
classes, probs, labels = predict_from_dataset('datasets/sam.txt', max_samples=100)
```

---

## Dataset Format

Input files should be CSV format with at least 3 columns:

```
sgRNA,DNA,label
GGTGAGTGAGTGTGTGCGTGTGG,TGTGAGTGTGTGTGTGTGTGTGT,1
GCCTCTTTCCCACCCACCTTGGG,GTCTCTTTCCCAGCGACCTGGGG,0
...
```

**Columns:**
- `sgRNA`: Guide RNA sequence
- `DNA`: Target DNA sequence
- `label`: Binary label (0 or 1)

---

## Model Architecture

### Input Processing

1. **CNN Encoding:** Sequence pairs encoded as (26, 7) matrix
   - 26 positions (23 bp + PAM)
   - 7 features per position (nucleotide matching)

2. **BERT Encoding:** Token-based representation
   - Token IDs (26 positions)
   - Segment IDs (sgRNA vs DNA)
   - Position IDs (0-25)

### Network Architecture

```
┌─────────────────────────────────────────┐
│  CNN Branch (20%)    BERT Branch (80%)  │
│                                          │
│  Inception CNN       Transformer        │
│  (26×7) → (26×80)    Layers             │
│                      Token + Segment +  │
│                      Position Embeddings│
│                      → (26×80)          │
│                                          │
│  BiGRU (20+20)       BiGRU (20+20)      │
│  → (26×40)           → (26×40)          │
│                                          │
│  Take last           Take last          │
│  timestep            timestep           │
│  → (40)              → (40)             │
│                                          │
│  × 0.2               × 0.8              │
└─────────────────────────────────────────┘
                  ↓
         Concatenate → (80)
                  ↓
          Dropout (0.35)
                  ↓
          Dense (128, ReLU)
                  ↓
          Dropout (0.35)
                  ↓
          Dense (64, ReLU)
                  ↓
          Dense (2, Softmax)
```

### Key Parameters

- **Vocab Size:** 28 tokens
- **BERT Embedding:** 256 dimensions
- **Attention Heads:** 4
- **Transformer Layers:** 2
- **Feed-Forward:** 1024 dimensions
- **Dropout Rate:** 0.35

---

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Adam optimizer |
| Batch Size | 256 | Large batch for stable gradients |
| Epochs | 30 | With early stopping (patience=5) |
| Loss Function | Sparse Categorical Crossentropy | Binary classification |
| Class Weighting | Auto | Handles imbalanced datasets |

### Adaptive Threshold

The model uses adaptive threshold optimization:
- Evaluates thresholds from 0.3 to 0.8
- Selects threshold maximizing F1 score
- Saves schedule to `weight/threshold_schedule.json`
- Applied automatically during inference

---

## Performance Metrics

The model reports comprehensive metrics:

- **AUROC** - Overall discrimination ability (0-1, higher is better)
- **PRAUC** - Precision-Recall AUC (handles class imbalance)
- **F1 Score** - Harmonic mean of precision and recall
- **MCC** - Matthews Correlation Coefficient (-1 to 1)
- **Accuracy** - Overall correctness
- **Precision** - Positive predictive value
- **Recall** - Sensitivity (true positive rate)
- **Specificity** - True negative rate

---

## Advanced Usage

### Training on Multiple Datasets

```python
from train_model import train_crispr_bert

model, history, metrics = train_crispr_bert(
    datasets=['datasets/I1.txt', 'datasets/I2.txt', 'datasets/II4.txt'],
    epochs=30,
    batch_size=256,
    learning_rate=1e-4,
    patience=5
)
```

### Custom Class Weights

```python
# For highly imbalanced datasets
class_weights = {0: 1.0, 1: 5.0}  # Weight class 1 more heavily

model, history, metrics = train_crispr_bert(
    datasets=['datasets/imbalanced.txt'],
    class_weights=class_weights,
    epochs=30
)
```

### Loading Specific Model

```python
from run_model import predict_single_sequence

# Use a specific checkpoint
pred_class, probs = predict_single_sequence(
    sgrna="GGTGAGTGAGTGTGTGCGTGTGG",
    dna="TGTGAGTGTGTGTGTGTGTGTGT",
    model_path='weight/best_model.h5',
    use_threshold=True
)
```

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model not found at 'weight/final_model.h5'
```

**Solution:** Train the model first:
```bash
python train_model.py
```

### Memory Issues

If you encounter OOM errors:
1. Reduce batch size: `batch_size=128` or `64`
2. Limit dataset: `load_dataset('file.txt', max_samples=10000)`
3. Use CPU instead of GPU (slower but more memory)

### Low Performance

If validation metrics are poor:
1. Train longer: Increase `epochs` or reduce `patience`
2. Adjust class weights for imbalanced data
3. Ensure dataset quality (correct labels, sufficient samples)
4. Try different learning rates: `1e-3`, `1e-4`, `1e-5`

---

## File Outputs

### Training Outputs

- `weight/final_model.h5` - Final trained model
- `weight/best_model.h5` - Best validation checkpoint
- `weight/threshold_schedule.json` - Adaptive threshold history

### Threshold Schedule Format

```json
{
  "final_threshold": 0.550,
  "schedule": [
    {
      "epoch": 0,
      "threshold": 0.500,
      "best_f1": 0.7234
    },
    {
      "epoch": 1,
      "threshold": 0.550,
      "best_f1": 0.7456
    }
  ]
}
```

---

## Model Architecture Details

### CNN Branch (Inception Module)

Four parallel convolution paths:
- **1×1 Conv:** 20 filters (fine-grained features)
- **3×1 Conv:** 20 filters (local patterns)
- **5×1 Conv:** 20 filters (broader context)
- **MaxPool + 1×1:** 20 filters (downsampling + projection)

Total output: 80 channels

### BERT Branch (Transformer)

**Embedding Layer:**
- Token embeddings (28 vocab × 256 dim)
- Segment embeddings (2 segments × 256 dim)
- Position embeddings (26 positions × 256 dim)

**Transformer Blocks (×2):**
- Multi-head attention (4 heads, 64 dim each)
- Layer normalization
- Feed-forward network (256 → 1024 → 256)
- Residual connections

**Output Projection:**
- Dense layer (256 → 80 dim)

---

## Citation

If you use this code, please cite the original CRISPR-BERT paper:
```
[Citation information would go here]
```

---

## License

[Add license information]

---

## Contact & Support

For issues or questions, please open an issue in the project repository.

---

**Last Updated:** October 2025
