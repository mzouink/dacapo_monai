#!/usr/bin/env python3
"""
Basic example showing how to use dacapo-monai for medical imaging preprocessing.

This example demonstrates:
1. Loading medical imaging data
2. Creating a basic MONAI transform pipeline
3. Using it with dacapo_monai.iterable_dataset
4. Iterating through batches
"""

import numpy as np
import torch
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import MedicalImagingTransforms
from funlib.persistence.arrays.array import Array


def create_sample_data():
    """Create sample medical imaging data for demonstration."""
    # Simulate 3D medical imaging data (e.g., CT scan)
    data = np.random.randint(0, 1000, (200, 200, 200)).astype(np.int16)

    # Add some structure to make it more realistic
    # Create a spherical structure in the center
    center = np.array([100, 100, 100])
    coords = np.ogrid[:200, :200, :200]
    distances = np.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )
    sphere_mask = distances <= 50
    data[sphere_mask] += 500  # Brighten the sphere

    # Create an Array object
    medical_array = Array(
        data=data,
        offset=(0, 0, 0),
        voxel_size=(1.0, 1.0, 1.0),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
    )

    return medical_array


def main():
    print("🏥 DaCapo-MONAI Basic Medical Imaging Example")
    print("=" * 50)

    # Create sample data
    print("📊 Creating sample medical data...")
    medical_data = create_sample_data()
    print(f"   Data shape: {medical_data.shape}")
    print(f"   Data range: {medical_data.data.min()} - {medical_data.data.max()}")
    print(f"   Voxel size: {medical_data.voxel_size}")

    # Create MONAI transforms for medical imaging
    print("\n🔧 Creating MONAI transform pipeline...")
    transforms = MedicalImagingTransforms.basic_3d_preprocessing(
        keys=["image"],
        intensity_range=(0, 1500),  # Typical medical imaging range
        output_range=(0.0, 1.0),
        spatial_size=(64, 64, 64),  # Crop to smaller patches
        add_gaussian_noise=True,
        noise_std=0.05,
        rotation_prob=0.3,
    )

    print("   Transform pipeline created with:")
    print("   - Intensity normalization (0-1500 → 0.0-1.0)")
    print("   - Spatial cropping to 64³")
    print("   - Gaussian noise augmentation")
    print("   - Random rotation (30% probability)")

    # Create iterable dataset
    print("\n📦 Creating dacapo-monai dataset...")
    dataset = iterable_dataset(
        datasets={"image": medical_data},
        shapes={"image": (96, 96, 96)},  # Larger than crop to ensure data
        transforms=transforms,  # Use MONAI transforms directly!
    )
    print("   ✅ Dataset created successfully!")

    # Test the dataset
    print("\n🔄 Testing dataset iteration...")
    batch_iterator = iter(dataset)

    for i in range(3):
        print(f"\n   Batch {i+1}:")
        batch = next(batch_iterator)

        print(f"     Keys: {list(batch.keys())}")
        print(f"     Image shape: {batch['image'].shape}")
        print(f"     Image dtype: {batch['image'].dtype}")
        print(
            f"     Image range: {batch['image'].min():.3f} - {batch['image'].max():.3f}"
        )
        print(f"     Image mean: {batch['image'].mean():.3f}")
        print(f"     Metadata preserved: {'metadata' in batch}")

        # Verify the transforms worked
        assert batch["image"].shape == torch.Size([64, 64, 64]), "Spatial crop failed"
        assert (
            0.0 <= batch["image"].min() <= batch["image"].max() <= 1.0
        ), "Intensity normalization failed"
        assert batch["image"].dtype == torch.float32, "Wrong tensor type"

    print("\n✅ All tests passed!")
    print("\n💡 Key benefits demonstrated:")
    print("   - Direct MONAI transform integration")
    print("   - Automatic format handling (numpy → tensor)")
    print("   - Preserved DaCapo metadata system")
    print("   - Medical imaging specific preprocessing")
    print("   - Ready for PyTorch training loops")

    print(f"\n🎯 Example usage in training:")
    print(f"   from torch.utils.data import DataLoader")
    print(f"   dataloader = DataLoader(dataset, batch_size=4)")
    print(f"   for batch in dataloader:")
    print(f"       # Your training code here")
    print(f"       pass")


if __name__ == "__main__":
    main()
