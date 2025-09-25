# API Reference

Complete API documentation for all DaCapo-MONAI modules.

```{toctree}
:maxdepth: 2

dataset
transforms
utils
```

## Quick Reference

### Main Functions

```{eval-rst}
.. currentmodule:: dacapo_monai

.. autosummary::
   :toctree: generated/
   
   iterable_dataset
   PipelineDataset
```

### Transform Presets

```{eval-rst}
.. currentmodule:: dacapo_monai.transforms

.. autosummary::
   :toctree: generated/
   
   SSLTransforms
   MedicalImagingTransforms
   ContrastiveLearningTransforms
```

### Utilities

```{eval-rst}
.. currentmodule:: dacapo_monai.utils

.. autosummary::
   :toctree: generated/
   
   MonaiToDacapoAdapter
   add_channel_dim
   remove_channel_dim
```

## Module Overview

| Module | Description |
|--------|-------------|
| {doc}`dataset` | Core dataset classes with MONAI transform integration |
| {doc}`transforms` | Pre-configured transform presets for common use cases |
| {doc}`utils` | Format conversion utilities and adapters |