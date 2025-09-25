# Examples

Real-world examples and tutorials for using DaCapo-MONAI.

```{toctree}
:maxdepth: 2

basic_usage
medical_imaging
self_supervised_learning  
custom_transforms
migration_guide
```

## Overview

Learn DaCapo-MONAI through practical examples that demonstrate real-world usage patterns.

::::{grid} 2
:::{grid-item-card} 🚀 Basic Usage
:link: basic_usage
:link-type: doc

Get started with simple examples showing core functionality.
:::

:::{grid-item-card} 🏥 Medical Imaging
:link: medical_imaging
:link-type: doc

Medical imaging preprocessing and augmentation workflows.
:::

:::{grid-item-card} 🔬 Self-Supervised Learning
:link: self_supervised_learning
:link-type: doc

Contrastive learning and SSL transform examples.
:::

:::{grid-item-card} 🎨 Custom Transforms
:link: custom_transforms
:link-type: doc

Create and integrate your own custom MONAI transforms.
:::
::::

## Example Categories

### By Use Case
- **Medical Image Analysis**: Preprocessing, segmentation, classification
- **Self-Supervised Learning**: Contrastive learning, masked modeling
- **Data Augmentation**: Advanced augmentation strategies
- **Multi-Modal**: Handling different data types together

### By Complexity
- **Beginner**: Basic integration and simple workflows
- **Intermediate**: Custom transforms and advanced features
- **Advanced**: Performance optimization and complex pipelines

## Running the Examples

All examples are available in the `examples/` directory:

```bash
# Clone the repository
git clone https://github.com/dacapo-toolbox/dacapo-monai.git

# Install the package
pip install -e ".[examples]"

# Run examples
python examples/basic_medical_imaging.py
python examples/ssl_contrastive_learning.py
```

## Interactive Notebooks

Many examples are also available as Jupyter notebooks for interactive exploration:

- 📓 **Basic Usage**: [basic_usage.ipynb](notebooks/basic_usage.ipynb)
- 📓 **Medical Imaging**: [medical_imaging.ipynb](notebooks/medical_imaging.ipynb)  
- 📓 **SSL Tutorial**: [ssl_tutorial.ipynb](notebooks/ssl_tutorial.ipynb)

## Community Examples

Check out community-contributed examples:

- **3D Segmentation**: Advanced segmentation workflows
- **Multi-GPU Training**: Distributed training examples
- **Custom Datasets**: Working with your own data formats

Want to contribute an example? See our [contribution guidelines](../developer_guide/contributing.md)!