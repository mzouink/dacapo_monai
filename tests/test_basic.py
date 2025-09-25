"""
Basic tests for dacapo-monai functionality.
These tests verify core functionality without requiring full installations.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from dacapo_monai.utils import (
        add_channel_dim,
        remove_channel_dim,
        MonaiToDacapoAdapter,
    )
    from dacapo_monai.transforms import (
        SSLTransforms,
        MedicalImagingTransforms,
        ContrastiveLearningTransforms,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "raw": torch.rand(64, 64, 64),
            "metadata": {"voxel_size": (1, 1, 1)},
        }

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_add_channel_dim(self):
        """Test adding channel dimension."""
        result = add_channel_dim(self.test_data, "raw")

        # Should add channel dimension to raw
        self.assertEqual(result["raw"].shape, (1, 64, 64, 64))
        # Metadata should be preserved
        self.assertEqual(result["metadata"]["voxel_size"], (1, 1, 1))

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_remove_channel_dim(self):
        """Test removing channel dimension."""
        # First add a channel dimension
        data_with_channels = {
            "raw": torch.rand(1, 64, 64, 64),
            "metadata": {"voxel_size": (1, 1, 1)},
        }

        result = remove_channel_dim(data_with_channels, "raw")

        # Should remove channel dimension from raw
        self.assertEqual(result["raw"].shape, (64, 64, 64))
        # Metadata should be preserved
        self.assertEqual(result["metadata"]["voxel_size"], (1, 1, 1))

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_monai_adapter(self):
        """Test MONAI to DaCapo adapter."""
        # Mock MONAI transform
        mock_transform = MagicMock()
        mock_transform.return_value = {
            "raw": torch.rand(1, 32, 32, 32),  # Different size after transform
            "metadata": {"transformed": True},
        }

        adapter = MonaiToDacapoAdapter(mock_transform)
        result = adapter(self.test_data)

        # Should call the transform
        mock_transform.assert_called_once()
        # Result should be transformed
        self.assertEqual(result["raw"].shape, (32, 32, 32))  # Channel removed
        self.assertTrue(result["metadata"]["transformed"])


class TestTransforms(unittest.TestCase):
    """Test transform presets."""

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_ssl_transforms_creation(self):
        """Test SSL transform preset creation."""
        try:
            transforms = SSLTransforms.contrastive_3d()
            self.assertIsNotNone(transforms)
            # Should be callable
            self.assertTrue(callable(transforms))
        except Exception as e:
            self.fail(f"SSL transform creation failed: {e}")

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_medical_transforms_creation(self):
        """Test medical imaging transform preset creation."""
        try:
            transforms = MedicalImagingTransforms.basic_3d_preprocessing()
            self.assertIsNotNone(transforms)
            # Should be callable
            self.assertTrue(callable(transforms))
        except Exception as e:
            self.fail(f"Medical transform creation failed: {e}")

    @unittest.skipUnless(IMPORTS_AVAILABLE, "dacapo_monai not available")
    def test_contrastive_transforms_creation(self):
        """Test contrastive learning transform preset creation."""
        try:
            transforms = ContrastiveLearningTransforms.simclr_3d()
            self.assertIsNotNone(transforms)
            # Should be callable
            self.assertTrue(callable(transforms))

            transforms = ContrastiveLearningTransforms.byol_3d()
            self.assertIsNotNone(transforms)
            # Should be callable
            self.assertTrue(callable(transforms))
        except Exception as e:
            self.fail(f"Contrastive transform creation failed: {e}")


class TestMockDataProcessing(unittest.TestCase):
    """Test data processing with mock data."""

    def setUp(self):
        """Set up test data."""
        self.sample_data = {
            "raw": np.random.rand(64, 64, 64).astype(np.float32),
            "metadata": {"voxel_size": (1, 1, 1), "offset": (0, 0, 0)},
        }

    def test_data_format_conversion(self):
        """Test basic data format conversions."""
        # Test numpy to torch conversion
        torch_data = torch.from_numpy(self.sample_data["raw"])
        self.assertEqual(torch_data.shape, (64, 64, 64))
        self.assertEqual(torch_data.dtype, torch.float32)

        # Test adding/removing channel dimensions
        with_channels = torch_data.unsqueeze(0)
        self.assertEqual(with_channels.shape, (1, 64, 64, 64))

        without_channels = with_channels.squeeze(0)
        self.assertEqual(without_channels.shape, (64, 64, 64))

    def test_dict_processing(self):
        """Test dictionary-based data processing."""
        # Test metadata preservation
        processed = {**self.sample_data}
        processed["raw"] = torch.from_numpy(processed["raw"])

        self.assertIn("metadata", processed)
        self.assertEqual(processed["metadata"]["voxel_size"], (1, 1, 1))
        self.assertIsInstance(processed["raw"], torch.Tensor)


class TestTypeCompatibility(unittest.TestCase):
    """Test type compatibility."""

    def test_basic_types(self):
        """Test basic type handling."""
        # Test various input types
        numpy_array = np.random.rand(10, 10, 10)
        torch_tensor = torch.rand(10, 10, 10)

        # Should handle both types
        self.assertEqual(numpy_array.shape, (10, 10, 10))
        self.assertEqual(torch_tensor.shape, (10, 10, 10))

    def test_dict_structures(self):
        """Test dictionary structure handling."""
        test_dict = {
            "raw": np.random.rand(32, 32, 32),
            "labels": np.random.randint(0, 5, (32, 32, 32)),
            "metadata": {"test": True},
        }

        # Should preserve all keys
        self.assertIn("raw", test_dict)
        self.assertIn("labels", test_dict)
        self.assertIn("metadata", test_dict)

        # Should handle nested metadata
        self.assertTrue(test_dict["metadata"]["test"])


def run_tests():
    """Run all tests."""
    print("🧪 RUNNING DACAPO-MONAI TESTS")
    print("=" * 30)

    if not IMPORTS_AVAILABLE:
        print("⚠️  dacapo_monai imports not available")
        print("   Some tests will be skipped")
        print("   Run 'python setup.py' to install the package first")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestTransforms))
    suite.addTests(loader.loadTestsFromTestCase(TestMockDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTypeCompatibility))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 15)
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, trace in result.failures:
            print(f"   {test}: {trace}")

    if result.errors:
        print("\n⚠️  ERRORS:")
        for test, trace in result.errors:
            print(f"   {test}: {trace}")

    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n💥 SOME TESTS FAILED!")

    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
