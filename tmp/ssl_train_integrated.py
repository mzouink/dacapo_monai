#!/usr/bin/env python3
"""
Example showing how to use MONAI transforms with dacapo toolbox for SSL training.
This replaces the separate train_transforms approach in ssl_train.py.
"""

import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from funlib.persistence import open_ds
from funlib.geometry import Coordinate

# MONAI imports
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    SpatialPadd,
    RandSpatialCropd,
    CopyItemsd,
    OneOf,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Lambda,
)

# DaCapo imports - fix import path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "dacapo-toolbox", "src"))
from dacapo_toolbox.dataset import iterable_dataset

# Setup
mp.set_start_method("fork", force=True)
import dask

dask.config.set(scheduler="single-threaded")


def main():
    # Load data
    data_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s0/"
    raw_train = open_ds(data_path)
    blocksize = Coordinate(256, 256, 256)

    print(f"Data shape: {raw_train.shape}, voxel size: {raw_train.voxel_size}")

    # Create MONAI transforms pipeline - integrated approach
    ssl_transforms = Compose(
        [
            # Convert to tensor and add channel dimension for MONAI compatibility
            Lambda(
                func=lambda x: {
                    **x,
                    "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float(),
                }
            ),
            # Scale intensity range
            ScaleIntensityRanged(
                keys=["raw"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Ensure minimum spatial size
            SpatialPadd(keys=["raw"], spatial_size=(96, 96, 96)),
            # Random spatial crop
            RandSpatialCropd(
                keys=["raw"],
                roi_size=(96, 96, 96),
                random_size=False,
            ),
            # Create copies for contrastive learning
            CopyItemsd(
                keys=["raw"],
                times=2,
                names=["gt_image", "image_2"],
                allow_missing_keys=False,
            ),
            # Apply augmentations to first copy (raw)
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
            # Apply different augmentations to second copy
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
            # Remove channel dimension for DaCapo compatibility
            Lambda(
                func=lambda x: {
                    k: v.squeeze(0) if k in ["raw", "gt_image", "image_2"] else v
                    for k, v in x.items()
                }
            ),
        ]
    )

    # Create dataset with integrated MONAI transforms
    train_dataset = iterable_dataset(
        datasets={"raw": raw_train},
        shapes={"raw": blocksize},
        transforms=ssl_transforms,  # Pass MONAI transforms directly!
        # You can still use DaCapo's geometric augmentations if needed
        # deform_augment_config=DeformAugmentConfig(...),
        # simple_augment_config=SimpleAugmentConfig(...),
    )

    # Test the dataset
    print("Testing integrated MONAI + DaCapo dataset...")
    batch_gen = iter(train_dataset)
    batch = next(batch_gen)

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Raw shape: {batch['raw'].shape}")
    print(f"GT image shape: {batch['gt_image'].shape}")
    print(f"Image 2 shape: {batch['image_2'].shape}")
    print(f"Raw value range: {batch['raw'].min():.3f} - {batch['raw'].max():.3f}")
    print(f"Metadata preserved: {'metadata' in batch}")

    # Visualize results (optional)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Show middle slices
        mid_z = batch["raw"].shape[0] // 2

        axes[0, 0].imshow(batch["raw"][mid_z].numpy(), cmap="gray")
        axes[0, 0].set_title("Augmented Raw")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(batch["gt_image"][mid_z].numpy(), cmap="gray")
        axes[0, 1].set_title("GT Image (Clean Copy)")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(batch["image_2"][mid_z].numpy(), cmap="gray")
        axes[1, 0].set_title("Image 2 (Different Augmentation)")
        axes[1, 0].axis("off")

        # Show difference
        diff = torch.abs(batch["raw"][mid_z] - batch["image_2"][mid_z])
        axes[1, 1].imshow(diff.numpy(), cmap="hot")
        axes[1, 1].set_title("Difference Map")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig("ssl_augmentation_example.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'ssl_augmentation_example.png'")

    except Exception as e:
        print(f"Visualization failed (non-critical): {e}")

    print("\n✅ MONAI + DaCapo integration working successfully!")
    print("\nKey benefits:")
    print("- Single unified transform pipeline")
    print("- Full MONAI transform library available")
    print("- Maintains DaCapo metadata and compatibility")
    print("- Can still use DaCapo geometric augmentations")
    print("- Cleaner, more maintainable code")


if __name__ == "__main__":
    main()
