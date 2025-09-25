# DaCapo-MONAI Documentation

Welcome to the official documentation for **DaCapo-MONAI**, a unified library that seamlessly integrates MONAI transforms with DaCapo toolbox datasets for medical image processing and self-supervised learning.

```{toctree}
:maxdepth: 2
:caption: Contents:

getting_started/index
user_guide/index
examples/index
api_reference/index
developer_guide/index
changelog
```

## Quick Links

::::{grid} 2
:::{grid-item-card} 🚀 Getting Started
:link: getting_started/index
:link-type: doc

New to DaCapo-MONAI? Start here for installation and basic usage.
:::

:::{grid-item-card} 📚 User Guide  
:link: user_guide/index
:link-type: doc

Comprehensive guides covering all features and capabilities.
:::

:::{grid-item-card} 💡 Examples
:link: examples/index
:link-type: doc

Real-world examples and tutorials for common use cases.
:::

:::{grid-item-card} 🔧 API Reference
:link: api_reference/index
:link-type: doc

Complete API documentation for all modules and classes.
:::
::::

## What is DaCapo-MONAI?

DaCapo-MONAI bridges the gap between **DaCapo toolbox** (large-scale 3D dataset processing) and **MONAI** (medical image transforms), providing:

- **🔄 Unified Transform Pipeline**: Use MONAI transforms directly in DaCapo datasets
- **🎯 Pre-configured Presets**: Expert-crafted transforms for SSL, medical imaging, and contrastive learning
- **⚡ Seamless Integration**: Drop-in replacement for existing DaCapo workflows
- **🧠 Type Safety**: Full type annotations and IDE support
- **📊 Format Handling**: Automatic conversion between DaCapo and MONAI data formats

## Quick Example

```python
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

# Create SSL transforms with one line
transforms = SSLTransforms.contrastive_3d()

# Use in your DaCapo dataset
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=transforms  # Full MONAI ecosystem available!
)

# Training loop stays the same
for batch in dataset:
    anchor = batch['raw']      # Original view
    positive = batch['raw_2']   # Augmented view
    # Your SSL training logic here
```

## Key Features

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - 🔄 **Unified Pipeline**
  - Single system supporting both DaCapo and MONAI transforms
* - 🎯 **Transform Presets** 
  - Expert-crafted compositions for SSL, medical imaging, contrastive learning
* - ⚡ **Drop-in Replacement**
  - Works with existing DaCapo workflows without changes
* - 🧠 **Type Safety**
  - Complete type annotations for better development experience
* - 📊 **Format Handling**
  - Automatic conversion between data formats
* - 🚀 **Performance**
  - Minimal overhead, maximum compatibility
```

## Installation

Install DaCapo-MONAI with pip:

```bash
pip install dacapo-monai
```

Or for development:

```bash
git clone https://github.com/dacapo-toolbox/dacapo-monai.git
cd dacapo-monai
pip install -e ".[dev]"
```

## Community

- 🐛 **Report Issues**: [GitHub Issues](https://github.com/dacapo-toolbox/dacapo-monai/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/dacapo-toolbox/dacapo-monai/discussions)
- 📧 **Support**: [support@dacapo-monai.org](mailto:support@dacapo-monai.org)

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`