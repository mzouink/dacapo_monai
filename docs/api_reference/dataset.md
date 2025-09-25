# Dataset Module

Core dataset functionality with MONAI transform support.

## Overview

The dataset module provides classes and functions for creating DaCapo-compatible datasets with MONAI transform integration.

## Key Components

### iterable_dataset
Main function for creating iterable datasets with MONAI transforms.

### PipelineDataset
Core dataset class that integrates MONAI transforms with DaCapo's data pipeline.

### Configuration Classes
- `SimpleAugmentConfig`: Basic augmentation configuration
- `DeformAugmentConfig`: Deformation-based augmentation configuration
- `MaskedSampling`: Masked sampling strategy
- `PointSampling`: Point-based sampling strategy

## API Reference

```{eval-rst}
.. automodule:: dacapo_monai.dataset
   :members:
   :undoc-members:
   :show-inheritance:
   :ignore-module-all:
```