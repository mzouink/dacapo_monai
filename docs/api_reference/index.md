````markdown
# API Reference

Complete API documentation for all DaCapo-MONAI modules.

```{toctree}
:maxdepth: 2

dataset
transforms
utils
```

## Module Overview

| Module | Description |
|--------|-------------|
| {doc}`dataset` | Core dataset classes with MONAI transform integration |
| {doc}`transforms` | Pre-configured transform presets for common use cases |
| {doc}`utils` | Format conversion utilities and adapters |

## Main Components

The DaCapo-MONAI library provides the following main components:

### Dataset Integration
- {doc}`generated/dacapo_monai.iterable_dataset`: Create iterable datasets with MONAI transform support
- {doc}`generated/dacapo_monai.PipelineDataset`: Core dataset class for DaCapo integration
- Configuration classes for different augmentation strategies

### Transform Presets
- `SSLTransforms`: Self-supervised learning transform configurations
- `MedicalImagingTransforms`: Common medical imaging preprocessing
- `ContrastiveLearningTransforms`: Transforms for contrastive learning

### Utilities
- Format conversion between MONAI and DaCapo data formats
- Metadata preservation helpers
- Channel dimension management utilities

For detailed documentation of each component, see the individual module pages.
````