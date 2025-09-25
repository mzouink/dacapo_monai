#!/usr/bin/env python3
"""
Complete demo script showing dacapo-monai functionality.
This script works with the current file structure and demonstrates all features.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, Callable, Union

# Add the src directory to the path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))


def create_mock_array():
    """Create a mock array object similar to funlib Array."""

    class MockArray:
        def __init__(self, data, offset, voxel_size):
            self.data = data
            self.offset = offset
            self.voxel_size = voxel_size
            self.shape = data.shape

        def to_ndarray(self):
            return self.data

        def numpy(self):
            return self.data

    data = np.random.rand(200, 200, 200).astype(np.float32)
    return MockArray(data, (0, 0, 0), (1, 1, 1))


def demo_basic_integration():
    """Demonstrate basic MONAI-DaCapo integration."""
    print("🔬 BASIC INTEGRATION DEMO")
    print("=" * 25)

    # Import our modules
    try:
        from dacapo_monai.utils import add_channel_dim, remove_channel_dim
        from dacapo_monai.transforms import SSLTransforms

        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Create test data
    test_data = {
        "raw": torch.rand(64, 64, 64),
        "metadata": {"voxel_size": (1, 1, 1), "offset": (0, 0, 0)},
    }

    print(f"📊 Original data shape: {test_data['raw'].shape}")

    # Test channel dimension utilities
    with_channels = add_channel_dim(test_data, "raw")
    print(f"📊 With channels: {with_channels['raw'].shape}")

    without_channels = remove_channel_dim(with_channels, "raw")
    print(f"📊 Without channels: {without_channels['raw'].shape}")

    # Test transform presets
    try:
        ssl_transforms = SSLTransforms.contrastive_3d()
        print("✅ SSL transforms created successfully")
        print(f"🔧 Transform type: {type(ssl_transforms)}")
    except Exception as e:
        print(f"⚠️  SSL transforms creation failed: {e}")

    print("✅ Basic integration demo completed!\n")
    return True


def demo_transform_presets():
    """Demonstrate transform presets."""
    print("🎯 TRANSFORM PRESETS DEMO")
    print("=" * 25)

    try:
        from dacapo_monai.transforms import (
            SSLTransforms,
            MedicalImagingTransforms,
            ContrastiveLearningTransforms,
        )

        print("🔬 SSL Transforms:")
        ssl_transforms = SSLTransforms.contrastive_3d()
        print(f"   ✅ Contrastive 3D: {type(ssl_transforms)}")

        print("\n🏥 Medical Imaging Transforms:")
        medical_transforms = MedicalImagingTransforms.basic_3d_preprocessing()
        print(f"   ✅ Basic 3D preprocessing: {type(medical_transforms)}")

        print("\n🎯 Contrastive Learning Transforms:")
        simclr_transforms = ContrastiveLearningTransforms.simclr_3d()
        print(f"   ✅ SimCLR 3D: {type(simclr_transforms)}")

        byol_transforms = ContrastiveLearningTransforms.byol_3d()
        print(f"   ✅ BYOL 3D: {type(byol_transforms)}")

        print("\n✅ All transform presets working!")

    except Exception as e:
        print(f"❌ Transform presets demo failed: {e}")
        return False

    print("✅ Transform presets demo completed!\n")
    return True


def demo_pipeline_dataset():
    """Demonstrate PipelineDataset functionality."""
    print("🏗️  PIPELINE DATASET DEMO")
    print("=" * 25)

    try:
        from dacapo_monai.dataset import PipelineDataset
        from dacapo_monai.transforms import SSLTransforms

        # Create mock data
        mock_array = create_mock_array()

        # Test with MONAI transforms
        ssl_transforms = SSLTransforms.contrastive_3d()

        # Create PipelineDataset
        dataset = PipelineDataset(
            datasets={"raw": mock_array},
            transforms=ssl_transforms,
            sample_shape={"raw": (64, 64, 64)},
        )

        print(f"✅ PipelineDataset created: {type(dataset)}")
        print(f"📊 Dataset shape: {dataset.sample_shape}")
        print(f"🔧 Transform type: {type(dataset.transforms)}")

        # Test one sample
        print("🧪 Testing sample generation...")
        sample = next(iter(dataset))
        print(f"✅ Sample generated successfully")
        print(f"📊 Sample keys: {list(sample.keys())}")

        for key, value in sample.items():
            if hasattr(value, "shape"):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")

        print("✅ Pipeline dataset demo completed!")

    except Exception as e:
        print(f"❌ Pipeline dataset demo failed: {e}")
        return False

    print("✅ Pipeline dataset demo completed!\n")
    return True


def demo_format_conversion():
    """Demonstrate format conversion utilities."""
    print("🔄 FORMAT CONVERSION DEMO")
    print("=" * 25)

    try:
        from dacapo_monai.utils import MonaiToDacapoAdapter

        # Mock MONAI transform
        class MockMonaiTransform:
            def __call__(self, data):
                # Simulate MONAI transform that adds channels and processes
                processed = {}
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        # Add channel dim and modify slightly
                        processed[key] = value.unsqueeze(0) * 0.9 + 0.1
                    else:
                        processed[key] = value
                return processed

        # Create adapter
        mock_transform = MockMonaiTransform()
        adapter = MonaiToDacapoAdapter(mock_transform)

        # Test data
        test_data = {"raw": torch.rand(32, 32, 32), "metadata": {"test": True}}

        print(f"📊 Input shape: {test_data['raw'].shape}")

        # Apply adapter
        result = adapter(test_data)

        print(f"📊 Output shape: {result['raw'].shape}")
        print(f"✅ Metadata preserved: {result.get('metadata', {}).get('test', False)}")

        print("✅ Format conversion demo completed!")

    except Exception as e:
        print(f"❌ Format conversion demo failed: {e}")
        return False

    print("✅ Format conversion demo completed!\n")
    return True


def demo_end_to_end_workflow():
    """Demonstrate complete end-to-end workflow."""
    print("🚀 END-TO-END WORKFLOW DEMO")
    print("=" * 27)

    try:
        from dacapo_monai.dataset import PipelineDataset
        from dacapo_monai.transforms import SSLTransforms

        # 1. Create data
        print("1️⃣  Creating sample data...")
        mock_array = create_mock_array()
        print(f"   ✅ Data shape: {mock_array.shape}")

        # 2. Create transforms
        print("2️⃣  Creating SSL transforms...")
        transforms = SSLTransforms.contrastive_3d()
        print("   ✅ Transforms ready")

        # 3. Create dataset
        print("3️⃣  Creating dataset...")
        dataset = PipelineDataset(
            datasets={"raw": mock_array},
            transforms=transforms,
            sample_shape={"raw": (64, 64, 64)},
        )
        print("   ✅ Dataset created")

        # 4. Generate training batches
        print("4️⃣  Generating training batches...")
        batch_count = 0
        for batch in dataset:
            batch_count += 1

            # Show what we got
            if batch_count == 1:
                print(f"   📊 Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if hasattr(value, "shape"):
                        print(f"      {key}: {value.shape}")
                    elif hasattr(value, "__len__") and not isinstance(value, str):
                        print(f"      {key}: length {len(value)}")
                    else:
                        print(f"      {key}: {type(value)}")

            # Stop after a few batches for demo
            if batch_count >= 3:
                break

        print(f"   ✅ Generated {batch_count} batches successfully")

        # 5. Simulate training step
        print("5️⃣  Simulating training step...")

        # Mock model
        class MockModel:
            def forward(self, x):
                return torch.randn(x.shape[0], 128)  # Mock embeddings

            def __call__(self, x):
                return self.forward(x)

        model = MockModel()

        # Get a batch
        sample_batch = next(iter(dataset))

        if "raw" in sample_batch:
            # Add batch dimension for model
            model_input = sample_batch["raw"].unsqueeze(0)
            output = model(model_input)
            print(f"   ✅ Model output shape: {output.shape}")

        print("✅ End-to-end workflow completed successfully!")

    except Exception as e:
        print(f"❌ End-to-end workflow demo failed: {e}")
        return False

    print("✅ End-to-end workflow demo completed!\n")
    return True


def run_all_demos():
    """Run all demonstration functions."""
    print("🎬 DACAPO-MONAI COMPLETE DEMO")
    print("=" * 32)
    print("Demonstrating full functionality without external dependencies")
    print()

    demos = [
        ("Basic Integration", demo_basic_integration),
        ("Transform Presets", demo_transform_presets),
        ("Pipeline Dataset", demo_pipeline_dataset),
        ("Format Conversion", demo_format_conversion),
        ("End-to-End Workflow", demo_end_to_end_workflow),
    ]

    results = {}

    for name, demo_func in demos:
        try:
            success = demo_func()
            results[name] = success
        except Exception as e:
            print(f"❌ {name} demo crashed: {e}")
            results[name] = False

    # Summary
    print("📋 DEMO RESULTS SUMMARY")
    print("=" * 23)

    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")

    total_passed = sum(results.values())
    total_demos = len(results)

    print(f"\n🎯 OVERALL: {total_passed}/{total_demos} demos passed")

    if total_passed == total_demos:
        print("\n🎉 ALL DEMOS SUCCESSFUL!")
        print("DaCapo-MONAI is working perfectly! 🚀")
        print("\n📚 NEXT STEPS:")
        print("1. Install with: python setup.py")
        print("2. Run examples: python examples/basic_medical_imaging.py")
        print("3. Read the docs: cat README.md")
        print("4. Start using in your projects!")
    else:
        print(f"\n⚠️  {total_demos - total_passed} demos failed")
        print("Some functionality may not be working correctly.")

    return total_passed == total_demos


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)
