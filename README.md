# Pneumonia X-Ray Classifier

Binary classification of chest X-rays as **Normal** or **Pneumonia** using a fine-tuned ResNet18 with transfer learning from ImageNet.

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate pneumonia-clf

# 2. Download data (Kaggle CLI)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# 3. Train
python -m src.train --lr 1e-4 --epochs 25 --seed 42

# 4. Evaluate
python -m src.evaluate --checkpoint checkpoints/best_model_seed42.pt

# 5. Run tests
pytest tests/ -v
```

## Project Structure

```
pneumonia-xray-classifier/
├── src/
│   ├── dataset.py      # Data loading, augmentation, weighted sampling
│   ├── model.py        # ResNet18 architecture with custom head
│   ├── train.py        # Training loop with AMP, early stopping, MLflow
│   └── evaluate.py     # Error analysis, confusion matrix, ROC curves
├── tests/
│   └── test_dataset.py # Unit tests for data pipeline
├── configs/
│   └── config.yaml     # Hyperparameters and paths
├── notebooks/
│   └── error_analysis.ipynb  # Interactive error analysis
├── requirements.txt
├── environment.yml
└── README.md
```

## Results

| Metric    | Value  |
|-----------|--------|
| Test F1   | 0.94   |
| AUROC     | 0.97   |
| Recall    | 0.96   |
| Precision | 0.92   |

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — CC BY 4.0 License

- 5,216 training images (1,341 Normal / 3,875 Pneumonia)
- 16 validation images
- 624 test images

## License

MIT License — See LICENSE file.
Code uses PyTorch (BSD) and torchvision (BSD).
