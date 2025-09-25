#!/usr/bin/env python3
"""
Self-supervised learning example using dacapo-monai.

This example demonstrates how to set up contrastive learning transforms
for self-supervised pretraining on 3D medical imaging data.

Based on approaches like:
- SimCLR: Simple Framework for Contrastive Learning of Visual Representations
- BYOL: Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
"""

import numpy as np
import torch
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms
from funlib.persistence.arrays.array import Array


def create_sample_medical_data():
    """Create sample 3D medical imaging data."""
    # Simulate larger medical volume
    data = np.random.randint(-100, 400, (400, 400, 400)).astype(np.int16)

    # Add anatomical structures
    # Large organ-like structure
    center1 = np.array([150, 200, 200])
    coords = np.ogrid[:400, :400, :400]
    distances1 = np.sqrt(
        (coords[0] - center1[0]) ** 2
        + (coords[1] - center1[1]) ** 2
        + (coords[2] - center1[2]) ** 2
    )
    organ_mask = distances1 <= 80
    data[organ_mask] += 200

    # Smaller structures
    center2 = np.array([250, 150, 250])
    distances2 = np.sqrt(
        (coords[0] - center2[0]) ** 2
        + (coords[1] - center2[1]) ** 2
        + (coords[2] - center2[2]) ** 2
    )
    small_structure = distances2 <= 30
    data[small_structure] += 300

    return Array(
        data=data,
        offset=(0, 0, 0),
        voxel_size=(2.0, 2.0, 2.0),  # 2nm voxel size
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
    )


def demonstrate_contrastive_learning():
    """Demonstrate contrastive learning setup."""
    print("🔬 Self-Supervised Contrastive Learning Example")
    print("=" * 55)

    # Create sample data
    print("📊 Creating sample medical volume...")
    medical_volume = create_sample_medical_data()
    print(f"   Volume shape: {medical_volume.shape}")
    print(
        f"   Intensity range: {medical_volume.data.min()} - {medical_volume.data.max()}"
    )
    print(f"   Voxel size: {medical_volume.voxel_size}")

    # Create SSL transforms for contrastive learning
    print("\n🔧 Setting up contrastive learning transforms...")
    ssl_transforms = SSLTransforms.contrastive_3d(
        keys=["raw"],
        intensity_range=(-100, 400),  # Match our data range
        output_range=(0.0, 1.0),
        spatial_size=(96, 96, 96),
        dropout_prob=0.8,
        shuffle_prob=0.6,
    )

    print("   Contrastive transform pipeline includes:")
    print("   - Intensity normalization")
    print("   - Random spatial cropping to 96³")
    print("   - Copy creation for contrastive pairs")
    print("   - Different augmentations per copy:")
    print("     * Coarse dropout (holes in data)")
    print("     * Coarse shuffle (local permutations)")
    print("   - Automatic channel dimension handling")

    # Create dataset
    print("\n📦 Creating SSL dataset...")
    dataset = iterable_dataset(
        datasets={"raw": medical_volume},
        shapes={"raw": (128, 128, 128)},  # Larger than crop size
        transforms=ssl_transforms,
    )

    # Test contrastive pairs
    print("\n🔄 Generating contrastive learning batches...")
    batch_iterator = iter(dataset)

    for i in range(2):
        print(f"\n   Contrastive Batch {i+1}:")
        batch = next(batch_iterator)

        print(f"     Available keys: {list(batch.keys())}")
        print(f"     Raw (augmented) shape: {batch['raw'].shape}")
        print(f"     GT (clean) shape: {batch['gt_raw'].shape}")
        print(f"     Second view shape: {batch['raw_2'].shape}")

        print(
            f"     Raw intensity: {batch['raw'].min():.3f} - {batch['raw'].max():.3f}"
        )
        print(
            f"     GT intensity: {batch['gt_raw'].min():.3f} - {batch['gt_raw'].max():.3f}"
        )
        print(
            f"     View 2 intensity: {batch['raw_2'].min():.3f} - {batch['raw_2'].max():.3f}"
        )

        # Calculate differences to show augmentations worked
        diff_1_vs_gt = torch.abs(batch["raw"] - batch["gt_raw"]).mean()
        diff_2_vs_gt = torch.abs(batch["raw_2"] - batch["gt_raw"]).mean()
        diff_1_vs_2 = torch.abs(batch["raw"] - batch["raw_2"]).mean()

        print(f"     Augmentation effects:")
        print(f"       Raw vs GT (clean): {diff_1_vs_gt:.4f}")
        print(f"       View2 vs GT: {diff_2_vs_gt:.4f}")
        print(f"       Raw vs View2: {diff_1_vs_2:.4f}")

    print("\n💡 Usage in contrastive learning:")
    print("   for batch in dataloader:")
    print("       anchor = batch['raw']      # Augmented view 1")
    print("       positive = batch['raw_2']   # Augmented view 2")
    print("       target = batch['gt_raw']    # Clean reference")
    print("       ")
    print("       # Encode views")
    print("       z_anchor = encoder(anchor)")
    print("       z_positive = encoder(positive)")
    print("       ")
    print("       # Contrastive loss")
    print("       loss = contrastive_loss(z_anchor, z_positive)")


def demonstrate_byol_setup():
    """Demonstrate BYOL (Bootstrap Your Own Latent) setup."""
    print("\n\n🎯 BYOL-Style Self-Supervised Learning")
    print("=" * 40)

    # Create data
    medical_volume = create_sample_medical_data()

    # Get BYOL transforms (returns tuple of online and target transforms)
    print("🔧 Creating BYOL transform pipelines...")
    from dacapo_monai.transforms import ContrastiveLearningTransforms

    online_transforms, target_transforms = ContrastiveLearningTransforms.byol_3d(
        keys=["raw"], crop_size=(64, 64, 64), intensity_range=(-100, 400)
    )

    print("   BYOL setup:")
    print("   - Online network: Strong augmentations")
    print("   - Target network: Weaker augmentations")
    print("   - No explicit negative samples needed")

    # Create datasets for both networks
    online_dataset = iterable_dataset(
        datasets={"raw": medical_volume},
        shapes={"raw": (96, 96, 96)},
        transforms=online_transforms,
    )

    target_dataset = iterable_dataset(
        datasets={"raw": medical_volume},
        shapes={"raw": (96, 96, 96)},
        transforms=target_transforms,
    )

    # Test both
    print("\n🔄 Testing BYOL data streams...")
    online_iter = iter(online_dataset)
    target_iter = iter(target_dataset)

    online_batch = next(online_iter)
    target_batch = next(target_iter)

    print(f"   Online batch shape: {online_batch['raw'].shape}")
    print(f"   Target batch shape: {target_batch['raw'].shape}")
    print(
        f"   Online intensity: {online_batch['raw'].min():.3f} - {online_batch['raw'].max():.3f}"
    )
    print(
        f"   Target intensity: {target_batch['raw'].min():.3f} - {target_batch['raw'].max():.3f}"
    )

    print("\n💡 BYOL training loop structure:")
    print("   for online_batch, target_batch in zip(online_loader, target_loader):")
    print("       # Online network forward")
    print("       online_proj = online_net(online_batch['raw'])")
    print("       online_pred = predictor(online_proj)")
    print("       ")
    print("       # Target network forward (no gradients)")
    print("       with torch.no_grad():")
    print("           target_proj = target_net(target_batch['raw'])")
    print("       ")
    print("       # BYOL loss (mean squared error)")
    print("       loss = F.mse_loss(online_pred, target_proj.detach())")


def main():
    """Run all SSL demonstrations."""
    try:
        demonstrate_contrastive_learning()
        demonstrate_byol_setup()

        print("\n\n✅ All SSL examples completed successfully!")
        print("\n🚀 Ready for self-supervised pretraining!")
        print("   Key advantages:")
        print("   - No manual data labeling required")
        print("   - Leverages large amounts of medical imaging data")
        print("   - Learns robust visual representations")
        print("   - Can be fine-tuned for downstream tasks")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install dacapo-monai[examples]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
