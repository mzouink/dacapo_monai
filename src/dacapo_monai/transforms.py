"""
Pre-configured transform compositions for common use cases.

This module provides ready-to-use MONAI transform compositions for:
- Self-supervised learning
- Medical imaging preprocessing
- Contrastive learning setups
"""

from typing import Any, Dict, Optional, Sequence, Union, List

try:
    from monai.transforms import (
        Compose,
        Lambda,
        ScaleIntensityRanged,
        SpatialPadd,
        RandSpatialCropd,
        CopyItemsd,
        OneOf,
        RandCoarseDropoutd,
        RandCoarseShuffled,
        RandGaussianNoised,
        RandRotated,
        RandAffined,
    )
except ImportError:
    raise ImportError(
        "MONAI is required for transform presets. Install with: pip install monai"
    )

import torch


class SSLTransforms:
    """Self-supervised learning transform presets."""

    @staticmethod
    def contrastive_3d(
        keys: Sequence[str] = ("raw",),
        intensity_range: tuple[float, float] = (-57, 164),
        output_range: tuple[float, float] = (0.0, 1.0),
        spatial_size: Sequence[int] = (96, 96, 96),
        dropout_prob: float = 0.8,
        shuffle_prob: float = 0.8,
    ) -> Compose:
        """
        Create transforms for 3D contrastive self-supervised learning.

        This preset creates augmented versions of input data suitable for
        contrastive learning approaches like SimCLR, BYOL, etc.

        Args:
            keys: Keys to process
            intensity_range: Input intensity range (a_min, a_max)
            output_range: Output intensity range (b_min, b_max)
            spatial_size: Target spatial size after cropping/padding
            dropout_prob: Probability of applying coarse dropout
            shuffle_prob: Probability of applying coarse shuffle

        Returns:
            MONAI Compose transform for contrastive learning
        """
        return Compose(
            [
                # Add channel dimension for MONAI compatibility
                Lambda(
                    func=lambda x: {
                        **x,
                        **{
                            k: torch.from_numpy(x[k].numpy()).unsqueeze(0).float()
                            for k in keys
                        },
                    }
                ),
                # Scale intensity
                ScaleIntensityRanged(
                    keys=list(keys),
                    a_min=intensity_range[0],
                    a_max=intensity_range[1],
                    b_min=output_range[0],
                    b_max=output_range[1],
                    clip=True,
                ),
                # Ensure minimum size
                SpatialPadd(keys=list(keys), spatial_size=spatial_size),
                # Random spatial crop
                RandSpatialCropd(
                    keys=list(keys),
                    roi_size=spatial_size,
                    random_size=False,
                ),
                # Create copies for contrastive learning
                CopyItemsd(
                    keys=list(keys),
                    times=2,
                    names=[f"gt_{k}" for k in keys] + [f"{k}_2" for k in keys],
                    allow_missing_keys=False,
                ),
                # Apply augmentations to first copy
                OneOf(
                    transforms=[
                        RandCoarseDropoutd(
                            keys=list(keys),
                            prob=dropout_prob,
                            holes=6,
                            spatial_size=5,
                            dropout_holes=True,
                            max_spatial_size=32,
                        ),
                        RandCoarseDropoutd(
                            keys=list(keys),
                            prob=dropout_prob,
                            holes=6,
                            spatial_size=20,
                            dropout_holes=False,
                            max_spatial_size=64,
                        ),
                    ]
                ),
                RandCoarseShuffled(
                    keys=list(keys), prob=shuffle_prob, holes=10, spatial_size=8
                ),
                # Apply different augmentations to second copy
                OneOf(
                    transforms=[
                        RandCoarseDropoutd(
                            keys=[f"{k}_2" for k in keys],
                            prob=dropout_prob,
                            holes=6,
                            spatial_size=5,
                            dropout_holes=True,
                            max_spatial_size=32,
                        ),
                        RandCoarseDropoutd(
                            keys=[f"{k}_2" for k in keys],
                            prob=dropout_prob,
                            holes=6,
                            spatial_size=20,
                            dropout_holes=False,
                            max_spatial_size=64,
                        ),
                    ]
                ),
                RandCoarseShuffled(
                    keys=[f"{k}_2" for k in keys],
                    prob=shuffle_prob,
                    holes=10,
                    spatial_size=8,
                ),
                # Remove channel dimension for DaCapo compatibility
                Lambda(
                    func=lambda x: {
                        k: (
                            v.squeeze(0)
                            if isinstance(v, torch.Tensor) and v.dim() > 3
                            else v
                        )
                        for k, v in x.items()
                    }
                ),
            ]
        )


class MedicalImagingTransforms:
    """Medical imaging preprocessing transforms."""

    @staticmethod
    def basic_3d_preprocessing(
        keys: Sequence[str] = ("image",),
        intensity_range: Optional[tuple[float, float]] = None,
        output_range: tuple[float, float] = (0.0, 1.0),
        spatial_size: Optional[Sequence[int]] = None,
        add_gaussian_noise: bool = True,
        noise_std: float = 0.1,
        rotation_prob: float = 0.5,
    ) -> Compose:
        """
        Basic 3D medical imaging preprocessing pipeline.

        Args:
            keys: Keys to process
            intensity_range: Input intensity range (a_min, a_max). If None, no scaling applied
            output_range: Output intensity range (b_min, b_max)
            spatial_size: Target spatial size. If None, no cropping/padding applied
            add_gaussian_noise: Whether to add Gaussian noise
            noise_std: Standard deviation for Gaussian noise
            rotation_prob: Probability of applying random rotation

        Returns:
            MONAI Compose transform for medical imaging
        """
        transforms: List[Any] = []

        # Add channel dimension
        transforms.append(
            Lambda(
                func=lambda x: {
                    **x,
                    **{
                        k: torch.from_numpy(x[k].numpy()).unsqueeze(0).float()
                        for k in keys
                    },
                }
            )
        )

        # Intensity scaling if specified
        if intensity_range is not None:
            transforms.append(
                ScaleIntensityRanged(
                    keys=list(keys),
                    a_min=intensity_range[0],
                    a_max=intensity_range[1],
                    b_min=output_range[0],
                    b_max=output_range[1],
                    clip=True,
                )
            )

        # Spatial operations if specified
        if spatial_size is not None:
            transforms.extend(
                [
                    SpatialPadd(keys=list(keys), spatial_size=spatial_size),
                    RandSpatialCropd(
                        keys=list(keys),
                        roi_size=spatial_size,
                        random_center=True,
                    ),
                ]
            )

        # Add noise if specified
        if add_gaussian_noise:
            transforms.append(
                RandGaussianNoised(
                    keys=list(keys),
                    prob=0.5,
                    mean=0.0,
                    std=noise_std,
                )
            )

        # Add rotation if specified
        if rotation_prob > 0:
            transforms.append(
                RandRotated(
                    keys=list(keys),
                    prob=rotation_prob,
                    range_x=0.1,
                    range_y=0.1,
                    range_z=0.1,
                    mode="bilinear",
                    padding_mode="border",
                )
            )

        # Remove channel dimension
        transforms.append(
            Lambda(
                func=lambda x: {
                    k: (
                        v.squeeze(0)
                        if isinstance(v, torch.Tensor) and v.dim() > 3
                        else v
                    )
                    for k, v in x.items()
                }
            )
        )

        return Compose(transforms)


class ContrastiveLearningTransforms:
    """Specialized transforms for contrastive learning setups."""

    @staticmethod
    def simclr_3d(
        keys: Sequence[str] = ("image",),
        crop_size: Sequence[int] = (64, 64, 64),
        intensity_range: tuple[float, float] = (0, 255),
        augmentation_strength: float = 0.5,
    ) -> Compose:
        """
        SimCLR-style transforms for 3D data.

        Args:
            keys: Keys to process
            crop_size: Size for random cropping
            intensity_range: Input intensity range
            augmentation_strength: Strength of augmentations (0.0 to 1.0)

        Returns:
            MONAI Compose transform for SimCLR
        """
        return Compose(
            [
                # Add channel dimension
                Lambda(
                    func=lambda x: {
                        **x,
                        **{
                            k: torch.from_numpy(x[k].numpy()).unsqueeze(0).float()
                            for k in keys
                        },
                    }
                ),
                # Intensity normalization
                ScaleIntensityRanged(
                    keys=list(keys),
                    a_min=intensity_range[0],
                    a_max=intensity_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Random resized crop equivalent
                RandSpatialCropd(
                    keys=list(keys),
                    roi_size=crop_size,
                    random_center=True,
                ),
                # Random affine transformation
                RandAffined(
                    keys=list(keys),
                    prob=augmentation_strength,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    translate_range=(10, 10, 10),
                    mode="bilinear",
                    padding_mode="border",
                ),
                # Color jittering equivalent - Gaussian noise
                RandGaussianNoised(
                    keys=list(keys),
                    prob=augmentation_strength * 0.8,
                    mean=0.0,
                    std=0.1,
                ),
                # Random coarse dropout (equivalent to random erasing)
                RandCoarseDropoutd(
                    keys=list(keys),
                    prob=augmentation_strength * 0.5,
                    holes=5,
                    spatial_size=8,
                    dropout_holes=True,
                    max_spatial_size=16,
                ),
                # Remove channel dimension
                Lambda(
                    func=lambda x: {
                        k: (
                            v.squeeze(0)
                            if isinstance(v, torch.Tensor) and v.dim() > 3
                            else v
                        )
                        for k, v in x.items()
                    }
                ),
            ]
        )

    @staticmethod
    def byol_3d(
        keys: Sequence[str] = ("image",),
        crop_size: Sequence[int] = (96, 96, 96),
        intensity_range: tuple[float, float] = (0, 255),
        return_both: bool = False,
    ) -> Union[Compose, tuple[Compose, Compose]]:
        """
        BYOL-style transforms for 3D data.

        Returns two different augmentation pipelines for the online and target networks.

        Args:
            keys: Keys to process
            crop_size: Size for random cropping
            intensity_range: Input intensity range
            return_both: If True, returns tuple of (online_transforms, target_transforms).
                        If False, returns only online_transforms for simple usage.

        Returns:
            Single Compose transform (online) or tuple of (online_transforms, target_transforms)
        """
        base_transforms: List[Any] = [
            # Add channel dimension
            Lambda(
                func=lambda x: {
                    **x,
                    **{
                        k: torch.from_numpy(x[k].numpy()).unsqueeze(0).float()
                        for k in keys
                    },
                }
            ),
            # Intensity normalization
            ScaleIntensityRanged(
                keys=list(keys),
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Random crop
            RandSpatialCropd(
                keys=list(keys),
                roi_size=crop_size,
                random_center=True,
            ),
        ]

        # Stronger augmentations for online network
        online_transforms = base_transforms + [
            RandAffined(
                keys=list(keys),
                prob=0.8,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.2, 0.2, 0.2),
                translate_range=(20, 20, 20),
                mode="bilinear",
                padding_mode="border",
            ),
            RandGaussianNoised(
                keys=list(keys),
                prob=0.6,
                mean=0.0,
                std=0.1,
            ),
            RandCoarseDropoutd(
                keys=list(keys),
                prob=0.5,
                holes=8,
                spatial_size=10,
                dropout_holes=True,
                max_spatial_size=20,
            ),
            # Remove channel dimension
            Lambda(
                func=lambda x: {
                    k: (
                        v.squeeze(0)
                        if isinstance(v, torch.Tensor) and v.dim() > 3
                        else v
                    )
                    for k, v in x.items()
                }
            ),
        ]

        # Weaker augmentations for target network
        target_transforms = base_transforms + [
            RandAffined(
                keys=list(keys),
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                mode="bilinear",
                padding_mode="border",
            ),
            # Remove channel dimension
            Lambda(
                func=lambda x: {
                    k: (
                        v.squeeze(0)
                        if isinstance(v, torch.Tensor) and v.dim() > 3
                        else v
                    )
                    for k, v in x.items()
                }
            ),
        ]

        online_compose = Compose(online_transforms)
        target_compose = Compose(target_transforms)

        if return_both:
            return online_compose, target_compose
        else:
            # Return just the online transforms for simple testing
            return online_compose
