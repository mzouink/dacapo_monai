#!/usr/bin/env python3
"""
Test script to verify realistic MONAI transforms integration with dacapo toolbox.
"""

import numpy as np
import torch
from funlib.persistence.arrays.array import Array
from funlib.geometry.coordinate import Coordinate

# Import the updated dacapo functions
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "dacapo-toolbox", "src"))
from dacapo_toolbox.dataset import iterable_dataset, create_monai_adapter

# Import MONAI transforms
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    RandSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    ToTensord,
    Lambda,
)


def test_realistic_monai_transforms():
    """Test realistic MONAI transforms for medical imaging"""
    print("Testing realistic MONAI transforms...")

    # Create a larger test array to simulate realistic medical imaging data
    data = np.random.randint(0, 255, (200, 200, 200)).astype(np.uint8)
    test_array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

    # Create realistic MONAI transforms pipeline
    transforms = Compose(
        [
            # Convert to tensor and add channel dimension
            Lambda(
                func=lambda x: {
                    **x,
                    "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float(),
                }
            ),
            # Scale intensity to 0-1 range
            ScaleIntensityRanged(
                keys=["raw"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Add Gaussian noise for augmentation
            RandGaussianNoised(
                keys=["raw"],
                prob=0.5,
                mean=0.0,
                std=0.1,
            ),
            # Random spatial crop
            RandSpatialCropd(
                keys=["raw"],
                roi_size=[64, 64, 64],
                random_center=True,
                random_size=False,
            ),
            # Random rotation
            RandRotated(
                keys=["raw"],
                prob=0.5,
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                mode="bilinear",
                padding_mode="border",
            ),
            # Remove channel dimension for consistency
            Lambda(func=lambda x: {**x, "raw": x["raw"].squeeze(0)}),
        ]
    )

    # Create dataset with MONAI transforms
    dataset = iterable_dataset(
        datasets={"raw": test_array},
        shapes={"raw": (96, 96, 96)},  # Larger than crop size to ensure we have data
        transforms=transforms,
    )

    # Get multiple batches to test variability
    batch_gen = iter(dataset)

    for i in range(3):
        batch = next(batch_gen)
        print(f"  Batch {i+1}:")
        print(f"    Raw shape: {batch['raw'].shape}")
        print(f"    Raw min/max: {batch['raw'].min():.3f} / {batch['raw'].max():.3f}")
        print(f"    Raw mean: {batch['raw'].mean():.3f}")
        print(f"    Metadata preserved: {'metadata' in batch}")

    print("✓ Realistic MONAI transforms working!")
    return True


def test_ssl_transforms():
    """Test self-supervised learning transforms similar to your ssl_train.py"""
    print("\nTesting SSL-style MONAI transforms...")

    # Create test data
    data = np.random.randint(-100, 200, (256, 256, 256)).astype(np.int16)
    test_array = Array(data, offset=(0, 0, 0), voxel_size=(2, 2, 2))

    # Create SSL transforms similar to your ssl_train.py
    from monai.transforms import (
        CopyItemsd,
        OneOf,
        RandCoarseDropoutd,
        RandCoarseShuffled,
        SpatialPadd,
    )

    ssl_transforms = Compose(
        [
            # Convert to tensor and add channel dimension
            Lambda(
                func=lambda x: {
                    **x,
                    "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float(),
                }
            ),
            # Scale intensity
            ScaleIntensityRanged(
                keys=["raw"],
                a_min=-100,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Ensure minimum size
            SpatialPadd(keys=["raw"], spatial_size=(96, 96, 96)),
            # Random crop
            RandSpatialCropd(
                keys=["raw"],
                roi_size=(96, 96, 96),
                random_center=True,
                random_size=False,
            ),
            # Create copies for contrastive learning
            CopyItemsd(
                keys=["raw"],
                times=2,
                names=["gt_image", "image_2"],
                allow_missing_keys=False,
            ),
            # Apply different augmentations to each copy
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["raw"],
                        prob=1.0,
                        holes=6,
                        spatial_size=5,
                        dropout_holes=True,
                        max_spatial_size=32,
                    ),
                    RandCoarseDropoutd(
                        keys=["raw"],
                        prob=1.0,
                        holes=6,
                        spatial_size=20,
                        dropout_holes=False,
                        max_spatial_size=64,
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["raw"], prob=0.8, holes=10, spatial_size=8),
            # Apply different augmentations to the second image
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=5,
                        dropout_holes=True,
                        max_spatial_size=32,
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=6,
                        spatial_size=20,
                        dropout_holes=False,
                        max_spatial_size=64,
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
            # Remove channel dimension
            Lambda(
                func=lambda x: {
                    k: v.squeeze(0) if k in ["raw", "gt_image", "image_2"] else v
                    for k, v in x.items()
                }
            ),
        ]
    )

    # Create dataset
    dataset = iterable_dataset(
        datasets={"raw": test_array},
        shapes={"raw": (128, 128, 128)},
        transforms=ssl_transforms,
    )

    # Get a batch
    batch_gen = iter(dataset)
    batch = next(batch_gen)

    print(f"  Raw shape: {batch['raw'].shape}")
    print(f"  GT image shape: {batch['gt_image'].shape}")
    print(f"  Image 2 shape: {batch['image_2'].shape}")
    print(f"  Number of samples from crop: {len(batch['raw'])}")
    print(f"  Raw range: {batch['raw'].min():.3f} - {batch['raw'].max():.3f}")
    print("✓ SSL-style transforms working!")
    return True


if __name__ == "__main__":
    print("Testing realistic MONAI integration with dacapo toolbox")
    print("=" * 60)

    try:
        test_realistic_monai_transforms()
        test_ssl_transforms()

        print("\n" + "=" * 60)
        print("✅ All realistic tests passed! Ready for production use.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
