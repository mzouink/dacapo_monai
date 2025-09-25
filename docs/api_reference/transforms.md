# Transforms Module

Pre-configured transform presets for common use cases.

## Overview

The transforms module provides ready-to-use transform configurations for various machine learning scenarios, particularly in medical imaging and self-supervised learning.

## Available Transform Presets

### SSLTransforms
Self-supervised learning transforms including:
- Contrastive learning augmentations
- Masking strategies
- Color jittering and geometric transforms

### MedicalImagingTransforms  
Common medical imaging preprocessing including:
- Intensity normalization
- Spatial resizing and cropping
- Medical-specific augmentations

### ContrastiveLearningTransforms
Specialized transforms for contrastive learning:
- Multi-view generation
- Strong and weak augmentation pairs
- Consistency-preserving transforms

## API Reference

```{eval-rst}
.. automodule:: dacapo_monai.transforms
   :members:
   :undoc-members:
   :show-inheritance:
   :ignore-module-all:
```