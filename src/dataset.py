"""
Chest X-Ray Pneumonia Dataset
==============================
Custom PyTorch Dataset for the Kaggle Chest X-Ray Images (Pneumonia) dataset.
Handles loading, preprocessing, and augmentation of chest X-ray images.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
License: CC BY 4.0
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class ChestXRayDataset(Dataset):
    """
    Dataset class for Chest X-Ray Pneumonia detection.

    The dataset is organized as:
        root/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/

    Labels: 0 = NORMAL, 1 = PNEUMONIA
    """

    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # Collect all image paths and labels
        self.samples = []
        self.labels = []

        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Directory not found: {class_dir}")

            for img_path in sorted(class_dir.glob("*.jpeg")):
                self.samples.append(img_path)
                self.labels.append(label)

        self.labels = np.array(self.labels)
        self.transform = self._build_transforms()

        # Log dataset statistics
        n_normal = (self.labels == 0).sum()
        n_pneumonia = (self.labels == 1).sum()
        print(f"[{split.upper()}] Loaded {len(self)} images: "
              f"{n_normal} normal, {n_pneumonia} pneumonia "
              f"(ratio: {n_pneumonia/max(n_normal,1):.2f})")

    def _build_transforms(self) -> transforms.Compose:
        """
        Build image transform pipeline.

        Training: Random augmentations to improve generalization.
        Validation/Test: Only resize and normalize (deterministic).

        WHY THESE AUGMENTATIONS:
        - RandomHorizontalFlip: X-rays can be mirrored without changing diagnosis.
        - RandomRotation(10°): Slight patient positioning variation is realistic.
        - ColorJitter(brightness): Accounts for exposure differences across machines.
        - NO vertical flip: Anatomically unrealistic for chest X-rays.
        """
        # ImageNet normalization — required for pretrained ResNet
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.augment and self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size + 32, self.img_size + 32)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image and convert to RGB (some X-rays are grayscale)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced dataset.

        The dataset has ~2.89x more pneumonia than normal images.
        Without weighting, the model would be biased toward predicting pneumonia.
        """
        counts = np.bincount(self.labels)
        weights = 1.0 / counts.astype(np.float32)
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance during training.

    Instead of class-weighted loss, this oversamples the minority class (NORMAL)
    so each batch has roughly equal class representation.
    """
    class_weights = dataset.get_class_weights()
    sample_weights = class_weights[dataset.labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def create_dataloaders(
    root_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Returns dict with keys: 'train', 'val', 'test'
    """
    # Set worker seed for reproducibility
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    # Create datasets
    train_dataset = ChestXRayDataset(root_dir, "train", img_size, augment=True)
    val_dataset = ChestXRayDataset(root_dir, "val", img_size, augment=False)
    test_dataset = ChestXRayDataset(root_dir, "test", img_size, augment=False)

    # Weighted sampler for training (handles class imbalance)
    train_sampler = get_weighted_sampler(train_dataset)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
