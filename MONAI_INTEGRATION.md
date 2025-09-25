# MONAI Integration with DaCapo Toolbox

This document explains how to use MONAI transforms with the enhanced DaCapo toolbox iterable dataset.

## Overview

The DaCapo toolbox `iterable_dataset` function now supports three types of transforms:

1. **DaCapo-style transforms**: Dictionary mapping transform signatures to callables
2. **MONAI-style transforms**: A single callable (like `monai.transforms.Compose`) that accepts and returns dictionaries
3. **MONAI adapter**: Helper function to wrap MONAI transforms for explicit compatibility

## Usage Examples

### 1. Original DaCapo-style Transforms (Still Supported)

```python
from dacapo_toolbox.dataset import iterable_dataset

dataset = iterable_dataset(
    datasets={"raw": my_array, "gt": my_gt},
    shapes={"raw": (128, 128, 128), "gt": (64, 64, 64)},
    transforms={
        ("raw", "processed_raw"): lambda x: x * 2.0,
        "gt": lambda x: x.float(),
        ("gt", "binary_mask"): lambda x: (x > 0).float()
    }
)
```

### 2. MONAI Transforms (Direct Usage)

```python
from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd
from dacapo_toolbox.dataset import iterable_dataset

# Create MONAI transform pipeline
monai_transforms = Compose([
    ScaleIntensityRanged(
        keys=["raw"],
        a_min=-57,
        a_max=164,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    RandSpatialCropd(
        keys=["raw"],
        roi_size=(96, 96, 96),
        random_center=True,
    ),
])

# Use directly with iterable_dataset
dataset = iterable_dataset(
    datasets={"raw": my_array},
    shapes={"raw": (128, 128, 128)},
    transforms=monai_transforms  # Pass MONAI transforms directly!
)
```

### 3. Self-Supervised Learning with MONAI

This example replicates the SSL transforms from your `ssl_train.py`:

```python
from monai.transforms import (
    Compose, ScaleIntensityRanged, SpatialPadd, RandSpatialCropd,
    CopyItemsd, OneOf, RandCoarseDropoutd, RandCoarseShuffled, Lambda
)
from dacapo_toolbox.dataset import iterable_dataset

# SSL transforms similar to your ssl_train.py
ssl_transforms = Compose([
    # Add channel dimension for MONAI compatibility
    Lambda(func=lambda x: {**x, "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float()}),
    
    # Scale intensity
    ScaleIntensityRanged(
        keys=["raw"],
        a_min=-57,
        a_max=164,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    
    # Ensure minimum size
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
    
    # Apply augmentations to first copy
    OneOf(transforms=[
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
    ]),
    
    RandCoarseShuffled(keys=["raw"], prob=0.8, holes=10, spatial_size=8),
    
    # Apply different augmentations to second copy
    OneOf(transforms=[
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
    ]),
    
    RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    
    # Remove channel dimension for consistency with DaCapo expectations
    Lambda(func=lambda x: {k: v.squeeze(0) if k in ["raw", "gt_image", "image_2"] else v 
                          for k, v in x.items()}),
])

# Use with iterable_dataset
dataset = iterable_dataset(
    datasets={"raw": raw_train},
    shapes={"raw": blocksize},
    transforms=ssl_transforms,  # MONAI transforms work directly!
    # You can still use other DaCapo features
    # deform_augment_config=DeformAugmentConfig(...),
    # simple_augment_config=SimpleAugmentConfig(...),
)
```

### 4. Using the MONAI Adapter (Optional)

For explicit compatibility or when you need more control:

```python
from dacapo_toolbox.dataset import create_monai_adapter

# Wrap MONAI transforms with the adapter
adapted_transforms = create_monai_adapter(monai_transforms)

dataset = iterable_dataset(
    datasets={"raw": my_array},
    shapes={"raw": (128, 128, 128)},
    transforms=adapted_transforms
)
```

## Key Features

1. **Backward Compatibility**: All existing DaCapo-style transforms continue to work
2. **Metadata Preservation**: MONAI transforms preserve the metadata that DaCapo needs
3. **Direct Integration**: No need for wrapper functions - pass MONAI `Compose` objects directly
4. **Flexible**: Mix and match with existing DaCapo augmentation configs

## Important Notes

### Tensor Format Considerations

MONAI expects tensors with channel dimensions, while DaCapo typically works without them. Use `Lambda` transforms to handle this:

```python
# Add channel dimension for MONAI processing
Lambda(func=lambda x: {**x, "raw": x["raw"].unsqueeze(0)}),

# Your MONAI transforms here...

# Remove channel dimension for DaCapo compatibility
Lambda(func=lambda x: {**x, "raw": x["raw"].squeeze(0)}),
```

### Key Naming

MONAI transforms work with the same keys as your datasets. If you have `{"raw": array}` in your dataset, use `keys=["raw"]` in MONAI transforms.

### Combining with DaCapo Augmentations

You can still use DaCapo's geometric augmentations alongside MONAI transforms:

```python
dataset = iterable_dataset(
    datasets={"raw": raw_train},
    shapes={"raw": blocksize},
    transforms=monai_transforms,  # MONAI transforms
    deform_augment_config=DeformAugmentConfig(...),  # DaCapo geometric transforms
    simple_augment_config=SimpleAugmentConfig(...),  # DaCapo simple transforms
)
```

The order of application is:
1. DaCapo deformation augmentations
2. DaCapo simple augmentations  
3. Your MONAI transforms

## Migration from ssl_train.py

To migrate your existing `ssl_train.py` to use this integrated approach:

1. Remove the separate `train_transforms` variable
2. Pass the MONAI `Compose` object directly to `transforms` parameter
3. Remove any manual tensor conversion - handle it within the MONAI pipeline
4. Test with a single batch to ensure expected behavior

The result will be cleaner code with the full power of MONAI transforms integrated seamlessly with DaCapo's data pipeline.