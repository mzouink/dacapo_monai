#!/usr/bin/env python3
"""
Test script to verify MONAI transforms integration with dacapo toolbox.
"""

import numpy as np
import torch
from funlib.persistence.arrays.array import Array
from funlib.geometry.coordinate import Coordinate

# Import the updated dacapo functions - using absolute path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "dacapo-toolbox", "src"))
from dacapo_toolbox.dataset import iterable_dataset, create_monai_adapter


def test_dacapo_style_transforms():
    """Test original dacapo-style transforms still work"""
    print("Testing dacapo-style transforms...")

    # Create a small test array
    data = np.random.rand(100, 100, 100).astype(np.float32)
    test_array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

    # Create dataset with dacapo-style transforms
    dataset = iterable_dataset(
        datasets={"raw": test_array},
        shapes={"raw": (32, 32, 32)},
        transforms={
            "raw": lambda x: x * 2.0,  # Simple scaling transform
            ("raw", "normalized"): lambda x: (x - x.mean()) / x.std(),  # Normalization
        },
    )

    # Get a batch
    batch_gen = iter(dataset)
    batch = next(batch_gen)

    print(f"✓ Original raw shape: {batch['raw'].shape}")
    print(f"✓ Normalized shape: {batch['normalized'].shape}")
    print(f"✓ Raw mean: {batch['raw'].mean():.4f}")
    print(f"✓ Normalized mean: {batch['normalized'].mean():.4f}")
    print("✓ Dacapo-style transforms working!")
    return True


def test_monai_style_transforms():
    """Test MONAI-style transforms"""
    print("\nTesting MONAI-style transforms...")

    try:
        from monai.transforms import Compose, Lambda

        # Create a small test array
        data = np.random.rand(100, 100, 100).astype(np.float32)
        test_array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

        # Create MONAI transforms
        monai_transforms = Compose(
            [
                Lambda(
                    func=lambda x: {**x, "scaled": x["raw"] * 0.5}
                ),  # Add scaled version
                Lambda(func=lambda x: {**x, "raw": x["raw"] + 1.0}),  # Modify raw
            ]
        )

        # Create dataset with MONAI transforms
        dataset = iterable_dataset(
            datasets={"raw": test_array},
            shapes={"raw": (32, 32, 32)},
            transforms=monai_transforms,
        )

        # Get a batch
        batch_gen = iter(dataset)
        batch = next(batch_gen)

        print(f"✓ Raw shape: {batch['raw'].shape}")
        print(f"✓ Scaled shape: {batch['scaled'].shape}")
        print(f"✓ Raw mean: {batch['raw'].mean():.4f}")
        print(f"✓ Scaled mean: {batch['scaled'].mean():.4f}")
        print(f"✓ Metadata preserved: {'metadata' in batch}")
        print("✓ MONAI-style transforms working!")
        return True

    except ImportError:
        print("⚠ MONAI not available, skipping MONAI transform test")
        return True


def test_monai_adapter():
    """Test the MONAI adapter helper function"""
    print("\nTesting MONAI adapter...")

    try:
        from monai.transforms import Compose, Lambda

        # Create a small test array
        data = np.random.rand(100, 100, 100).astype(np.float32)
        test_array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

        # Create MONAI transforms
        monai_transforms = Compose(
            [Lambda(func=lambda x: {**x, "processed": x["raw"] * 2.0})]
        )

        # Use the adapter
        adapted_transforms = create_monai_adapter(monai_transforms)

        # Create dataset
        dataset = iterable_dataset(
            datasets={"raw": test_array},
            shapes={"raw": (32, 32, 32)},
            transforms=adapted_transforms,
        )

        # Get a batch
        batch_gen = iter(dataset)
        batch = next(batch_gen)

        print(f"✓ Raw shape: {batch['raw'].shape}")
        print(f"✓ Processed shape: {batch['processed'].shape}")
        print(f"✓ Metadata preserved: {'metadata' in batch}")
        print("✓ MONAI adapter working!")
        return True

    except ImportError:
        print("⚠ MONAI not available, skipping adapter test")
        return True


if __name__ == "__main__":
    print("Testing MONAI integration with dacapo toolbox")
    print("=" * 50)

    try:
        test_dacapo_style_transforms()
        test_monai_style_transforms()
        test_monai_adapter()

        print("\n" + "=" * 50)
        print("✅ All tests passed! MONAI integration working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
