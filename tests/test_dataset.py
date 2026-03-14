"""
Unit & Integration Tests for Dataset Pipeline
================================================
Tests data loading, preprocessing, and augmentation logic.

Run: pytest tests/test_dataset.py -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import ChestXRayDataset, get_weighted_sampler


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary dataset directory with fake X-ray images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ["train", "val", "test"]:
            for cls in ["NORMAL", "PNEUMONIA"]:
                class_dir = Path(tmpdir) / split / cls
                class_dir.mkdir(parents=True)

                # Create 5 fake grayscale images per class
                n_images = 5 if cls == "NORMAL" else 15  # Imbalanced
                for i in range(n_images):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (256, 256), dtype=np.uint8),
                        mode="L",
                    )
                    img.save(class_dir / f"{cls.lower()}_{i:03d}.jpeg")

        yield tmpdir


class TestChestXRayDataset:
    """Tests for the ChestXRayDataset class."""

    def test_dataset_loads_correctly(self, mock_dataset_dir):
        """Verify dataset loads all images and assigns correct labels."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        assert len(dataset) == 20  # 5 normal + 15 pneumonia

    def test_labels_are_correct(self, mock_dataset_dir):
        """Verify label encoding: NORMAL=0, PNEUMONIA=1."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        assert (dataset.labels == 0).sum() == 5
        assert (dataset.labels == 1).sum() == 15

    def test_output_shape(self, mock_dataset_dir):
        """Verify output tensor shapes are correct for ResNet input."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        image, label = dataset[0]

        assert image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {image.shape}"
        assert isinstance(label, (int, np.integer))
        assert label in [0, 1]

    def test_grayscale_to_rgb_conversion(self, mock_dataset_dir):
        """Verify grayscale X-rays are converted to 3-channel RGB."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        image, _ = dataset[0]
        assert image.shape[0] == 3, "Grayscale should be converted to 3-channel RGB"

    def test_augmentation_is_stochastic(self, mock_dataset_dir):
        """Verify training augmentations produce different outputs."""
        dataset = ChestXRayDataset(
            mock_dataset_dir, split="train", img_size=224, augment=True
        )

        img1, _ = dataset[0]
        img2, _ = dataset[0]

        # With random augmentation, same index should produce different tensors
        # (probabilistically — very unlikely to be identical)
        assert not torch.allclose(img1, img2), \
            "Augmented images should differ across calls"

    def test_val_transforms_are_deterministic(self, mock_dataset_dir):
        """Verify validation transforms are deterministic (no randomness)."""
        dataset = ChestXRayDataset(
            mock_dataset_dir, split="val", img_size=224, augment=False
        )

        img1, _ = dataset[0]
        img2, _ = dataset[0]

        assert torch.allclose(img1, img2), \
            "Validation transforms should be deterministic"

    def test_class_weights_inversely_proportional(self, mock_dataset_dir):
        """Verify class weights handle imbalance correctly."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        weights = dataset.get_class_weights()

        assert weights[0] > weights[1], \
            "Normal (minority) class should have higher weight"
        assert len(weights) == 2

    def test_weighted_sampler_balances_classes(self, mock_dataset_dir):
        """Verify WeightedRandomSampler samples roughly equal classes."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="train", img_size=224)
        sampler = get_weighted_sampler(dataset)

        # Sample 200 indices and check balance
        sampled_labels = [dataset.labels[idx] for idx in list(sampler)[:200]]
        n_normal = sum(1 for l in sampled_labels if l == 0)
        n_pneumonia = sum(1 for l in sampled_labels if l == 1)

        ratio = n_normal / max(n_pneumonia, 1)
        assert 0.3 < ratio < 3.0, \
            f"Sampler should roughly balance classes, got ratio {ratio:.2f}"

    def test_missing_directory_raises_error(self, mock_dataset_dir):
        """Verify proper error on missing data directory."""
        with pytest.raises(FileNotFoundError):
            ChestXRayDataset("/nonexistent/path", split="train")


class TestImageNormalization:
    """Tests for image normalization values."""

    def test_imagenet_normalization(self, mock_dataset_dir):
        """Verify images are normalized with ImageNet statistics."""
        dataset = ChestXRayDataset(mock_dataset_dir, split="val", img_size=224)
        image, _ = dataset[0]

        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert image.min() >= -5.0, f"Min value {image.min()} seems wrong"
        assert image.max() <= 5.0, f"Max value {image.max()} seems wrong"
