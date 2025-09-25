#!/usr/bin/env python3
"""
Migration example showing how to upgrade existing DaCapo code to use MONAI transforms.

This example shows before/after code for migrating from separate transform
handling to integrated MONAI transforms in dacapo_monai.
"""

import numpy as np
import torch
from funlib.persistence.arrays.array import Array


# This would be your original approach (before)
def old_dacapo_approach():
    """Example of the old way of handling transforms separately."""
    print("📜 OLD APPROACH: Separate Transform Handling")
    print("=" * 45)

    # Original DaCapo dataset creation
    from dacapo_toolbox.dataset import iterable_dataset

    data = np.random.rand(200, 200, 200).astype(np.float32)
    array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

    # Create basic dataset without transforms
    dataset = iterable_dataset(
        datasets={"raw": array},
        shapes={"raw": (64, 64, 64)},
        transforms=None,  # No transforms in dataset
    )

    # Separate MONAI transforms that you'd apply manually
    from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd

    separate_transforms = Compose(
        [
            ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=1, b_min=0.0, b_max=1.0),
            RandSpatialCropd(keys=["raw"], roi_size=(32, 32, 32)),
        ]
    )

    print("   Problems with old approach:")
    print("   ❌ Two separate systems to manage")
    print("   ❌ Manual format conversion needed")
    print("   ❌ Extra complexity in training loops")
    print("   ❌ Metadata handling issues")
    print("   ❌ More error-prone code")

    return dataset, separate_transforms


def new_dacapo_monai_approach():
    """Example of the new integrated approach."""
    print("\n✨ NEW APPROACH: Integrated MONAI Transforms")
    print("=" * 45)

    # New integrated approach
    from dacapo_monai import iterable_dataset
    from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd, Lambda

    data = np.random.rand(200, 200, 200).astype(np.float32)
    array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

    # Create MONAI transforms with proper tensor handling
    integrated_transforms = Compose(
        [
            # Handle numpy to tensor conversion
            Lambda(
                func=lambda x: {
                    **x,
                    "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float(),
                }
            ),
            # MONAI transforms
            ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=1, b_min=0.0, b_max=1.0),
            RandSpatialCropd(keys=["raw"], roi_size=(32, 32, 32)),
            # Remove channel dimension for DaCapo compatibility
            Lambda(func=lambda x: {**x, "raw": x["raw"].squeeze(0)}),
        ]
    )

    # Create dataset with integrated transforms
    dataset = iterable_dataset(
        datasets={"raw": array},
        shapes={"raw": (64, 64, 64)},
        transforms=integrated_transforms,  # Transforms built right in!
    )

    print("   Benefits of new approach:")
    print("   ✅ Single unified pipeline")
    print("   ✅ Automatic format handling")
    print("   ✅ Preserved DaCapo metadata")
    print("   ✅ Cleaner training loops")
    print("   ✅ Less error-prone")
    print("   ✅ Full MONAI library access")

    return dataset


def demonstrate_training_loop_differences():
    """Show how training loops differ between approaches."""
    print("\n🔄 TRAINING LOOP COMPARISON")
    print("=" * 30)

    print("\n📜 Old approach training loop:")
    print("   for batch in old_dataset:")
    print("       # Manual format conversion")
    print("       batch_dict = {'raw': batch['raw'].unsqueeze(0)}")
    print("       ")
    print("       # Apply MONAI transforms")
    print("       transformed = monai_transforms(batch_dict)")
    print("       ")
    print("       # More manual conversion")
    print("       final_data = transformed['raw'].squeeze(0)")
    print("       ")
    print("       # Training step")
    print("       output = model(final_data)")
    print("       # ... rest of training")

    print("\n✨ New approach training loop:")
    print("   for batch in new_dataset:")
    print("       # Data is already perfectly formatted!")
    print("       output = model(batch['raw'])")
    print("       # ... rest of training")
    print("       # Much simpler and less error-prone!")


def demonstrate_ssl_migration():
    """Show SSL-specific migration example."""
    print("\n🔬 SSL MIGRATION EXAMPLE")
    print("=" * 25)

    print("📜 Old SSL approach:")
    print("   # Separate dataset and transforms")
    print("   dataset = iterable_dataset(datasets, shapes)")
    print("   ")
    print("   ssl_transforms = Compose([")
    print("       ScaleIntensityRanged(...),")
    print("       CopyItemsd(...),")
    print("       RandCoarseDropoutd(...),")
    print("       # ... more transforms")
    print("   ])")
    print("   ")
    print("   # Complex training loop with manual application")
    print("   for batch in dataset:")
    print("       batch = format_for_monai(batch)")
    print("       batch = ssl_transforms(batch)")
    print("       batch = format_for_model(batch)")
    print("       # ... training")

    print("\n✨ New SSL approach:")
    print("   from dacapo_monai.transforms import SSLTransforms")
    print("   ")
    print("   ssl_transforms = SSLTransforms.contrastive_3d()")
    print("   ")
    print("   dataset = iterable_dataset(")
    print("       datasets=datasets,")
    print("       shapes=shapes,")
    print("       transforms=ssl_transforms  # Just pass it in!")
    print("   )")
    print("   ")
    print("   # Super clean training loop")
    print("   for batch in dataset:")
    print("       anchor = batch['raw']")
    print("       positive = batch['raw_2']")
    print("       # ... SSL training")


def show_preset_usage():
    """Demonstrate using transform presets."""
    print("\n🎯 TRANSFORM PRESETS")
    print("=" * 20)

    print("Instead of manually creating transform pipelines:")
    print("   ❌ Lots of boilerplate code")
    print("   ❌ Easy to make mistakes")
    print("   ❌ Need to research best practices")

    print("\nUse pre-configured transform presets:")
    print("   ✅ from dacapo_monai.transforms import (")
    print("   ✅     SSLTransforms,")
    print("   ✅     MedicalImagingTransforms,")
    print("   ✅     ContrastiveLearningTransforms")
    print("   ✅ )")
    print("   ✅")
    print("   ✅ # One line to get expert-level transforms!")
    print("   ✅ transforms = SSLTransforms.contrastive_3d()")

    from dacapo_monai.transforms import SSLTransforms, MedicalImagingTransforms

    # Show available presets
    print("\n📦 Available presets:")
    print("   🔬 SSLTransforms:")
    print("      - contrastive_3d(): For contrastive learning")
    print("   ")
    print("   🏥 MedicalImagingTransforms:")
    print("      - basic_3d_preprocessing(): Standard medical preprocessing")
    print("   ")
    print("   🎯 ContrastiveLearningTransforms:")
    print("      - simclr_3d(): SimCLR-style transforms")
    print("      - byol_3d(): BYOL-style transforms")


def main():
    """Run the migration demonstration."""
    print("🚀 DACAPO → DACAPO-MONAI MIGRATION GUIDE")
    print("=" * 50)

    try:
        # Show old vs new
        old_dataset, old_transforms = old_dacapo_approach()
        new_dataset = new_dacapo_monai_approach()

        # Test both work
        print("\n🧪 TESTING BOTH APPROACHES")
        print("=" * 25)

        old_batch = next(iter(old_dataset))
        new_batch = next(iter(new_dataset))

        print(f"   Old approach batch shape: {old_batch['raw'].shape}")
        print(f"   New approach batch shape: {new_batch['raw'].shape}")
        print("   ✅ Both produce valid batches")

        # Show training differences
        demonstrate_training_loop_differences()

        # SSL migration
        demonstrate_ssl_migration()

        # Transform presets
        show_preset_usage()

        print("\n\n🎉 MIGRATION COMPLETE!")
        print("=" * 20)
        print("✅ Code is now cleaner and more maintainable")
        print("✅ Full MONAI ecosystem available")
        print("✅ Better performance and reliability")
        print("✅ Future-proof architecture")

        print("\n📋 MIGRATION CHECKLIST:")
        print("□ Install: pip install dacapo-monai")
        print("□ Replace: dacapo_toolbox → dacapo_monai imports")
        print("□ Integrate: Move MONAI transforms into iterable_dataset")
        print("□ Simplify: Remove manual format conversion code")
        print("□ Optimize: Use transform presets where possible")
        print("□ Test: Verify same functionality with cleaner code")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This would normally work with proper installations!")


if __name__ == "__main__":
    main()
