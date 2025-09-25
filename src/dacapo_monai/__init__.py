"""
DaCapo-MONAI Integration Library

This library provides seamless integration between MONAI transforms and DaCapo toolbox,
enabling the use of MONAI's extensive medical imaging transform library with DaCapo's
data pipeline infrastructure.

Key Features:
- Direct MONAI transform support in DaCapo iterable datasets
- Backward compatibility with existing DaCapo transforms
- Metadata preservation for proper DaCapo functionality
- Helper utilities for common use cases
- Self-supervised learning transform presets

Example:
    from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd
    from dacapo_monai import iterable_dataset

    transforms = Compose([
        ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
        RandSpatialCropd(keys=["raw"], roi_size=(96, 96, 96))
    ])

    dataset = iterable_dataset(
        datasets={"raw": my_array},
        shapes={"raw": (128, 128, 128)},
        transforms=transforms
    )
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .dataset import (
    iterable_dataset,
    PipelineDataset,
    create_monai_adapter,
    SimpleAugmentConfig,
    DeformAugmentConfig,
    MaskedSampling,
    PointSampling,
)

from .transforms import (
    SSLTransforms,
    MedicalImagingTransforms,
    ContrastiveLearningTransforms,
)

from .utils import (
    add_channel_dim,
    remove_channel_dim,
    ensure_tensor,
    preserve_metadata,
)

__all__ = [
    # Core functionality
    "iterable_dataset",
    "PipelineDataset",
    "create_monai_adapter",
    # Configuration classes
    "SimpleAugmentConfig",
    "DeformAugmentConfig",
    "MaskedSampling",
    "PointSampling",
    # Transform presets
    "SSLTransforms",
    "MedicalImagingTransforms",
    "ContrastiveLearningTransforms",
    # Utilities
    "add_channel_dim",
    "remove_channel_dim",
    "ensure_tensor",
    "preserve_metadata",
]
