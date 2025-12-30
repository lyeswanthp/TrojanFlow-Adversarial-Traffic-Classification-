# TrojanFlow: Adversarial Traffic Classification

Implementation of TrojanFlow backdoor attack against network traffic classifiers using deep neural networks.

## Overview

This project implements a backdoor attack on encrypted network traffic classification systems. The attack uses a trigger generator to create imperceptible perturbations that force a pre-trained classifier to misclassify traffic to a target class while maintaining high accuracy on clean samples.

## Architecture

**Trigger Generator**: Encoder-decoder network with class conditioning that generates flow-specific triggers based on preceding and current network flows.

**Traffic Classifier**: Multi-scale CNN with residual blocks for encrypted traffic classification using packet size sequences.

## Features

- Multi-scale convolutional feature extraction
- Class-conditional trigger generation
- Joint adversarial training with regularization
- Imperceptibility constraints (L1/L2 regularization)
- Automated evaluation and visualization

## Requirements

```
torch
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

## Dataset

Uses ISCXVPN2016 dataset with 9 high-performing application classes:
- FacebookVideo, Hangouts, SFTP, Torrent, SCP, Vimeo, Netflix, Spotify, YouTube

Preprocessed data expected in pickle format with packet size sequences (256 packets per flow).

## Usage

### Train Traffic Classifier

```bash
python improved_classifier.py
```

Trains a clean CNN classifier on network traffic flows.

### Launch TrojanFlow Attack

```bash
python trojanflow_implementation.py
```

Performs the backdoor attack through:
1. Pre-training the trigger generator
2. Joint training of generator and classifier
3. Evaluation on test set with confusion matrices
4. Trigger visualization

### SLURM Cluster

```bash
sbatch sh_script.sh
```

## Training Pipeline

1. **Classifier Pre-training**: Train on clean data to establish baseline performance
2. **Generator Pre-training**: Train trigger generator to maximize attack success rate
3. **Joint Training**: Adversarially train both networks with balanced objectives:
   - Maximize attack success rate on poisoned samples
   - Maintain accuracy on clean samples
   - Minimize trigger magnitude (imperceptibility)

## Evaluation Metrics

- **Clean Accuracy**: Classification accuracy on unmodified traffic
- **Attack Success Rate (ASR)**: Percentage of poisoned samples classified as target class
- **Confusion Matrices**: Visual comparison of clean vs. poisoned predictions

## Results

The trained models generate visualizations:
- `trojanflow_training_history.png`: Training curves
- `clean_confusion_matrix.png`: Performance on clean traffic
- `poisoned_confusion_matrix.png`: Performance on backdoored traffic
- `trojanflow_trigger_visualization.png`: Generated trigger patterns

## Model Checkpoints

Best models are saved automatically based on balanced scoring:
- `trojanflow_best_*.pt`: Best performing checkpoint
- `trojanflow_final.pt`: Final model state

## Implementation Details

- **Optimizer**: AdamW with cosine annealing
- **Regularization**: L1 (sparsity) + L2 (magnitude) + MSE (imperceptibility)
- **Batch Size**: 64
- **Poison Ratio**: Gradually increased to 50%
- **Gradient Clipping**: Applied for training stability

## License

Research and educational purposes only.

## Citation

Implementation based on adversarial machine learning techniques for network traffic classification.
