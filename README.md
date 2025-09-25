# DaCapo-MONAI

A unified library that seamlessly integrates MONAI transforms with DaCapo toolbox datasets for medical image processing and self-supervised learning.

[![PyPI version](https://badge.fury.io/py/dacapo-monai.svg)](https://badge.fury.io/py/dacapo-monai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 What is DaCapo-MONAI?

DaCapo-MONAI bridges the gap between **DaCapo toolbox** (large-scale 3D dataset processing) and **MONAI** (medical image transforms), providing:

- **🔄 Unified Transform Pipeline**: Use MONAI transforms directly in DaCapo datasets
- **🎯 Pre-configured Presets**: Expert-crafted transforms for SSL, medical imaging, and contrastive learning
- **⚡ Seamless Integration**: Drop-in replacement for existing DaCapo workflows
- **🧠 Type Safety**: Full type annotations and IDE support
- **📊 Format Handling**: Automatic conversion between DaCapo and MONAI data formats

## 🛠️ Installation

```bash
pip install dacapo-monai
```

### Dependencies

- `torch >= 1.10.0`
- `monai >= 1.0.0`
- `dacapo-toolbox >= 0.1.0`
- `gunpowder >= 1.3.0`
- `funlib.geometry >= 0.1.0`
- `funlib.persistence >= 1.0.0`

## 🏃‍♀️ Quick Start

### Basic Usage

Replace your existing DaCapo dataset creation:

```python
# Before (DaCapo only)
from dacapo_toolbox.dataset import iterable_dataset

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=None  # Limited transform options
)

# After (DaCapo-MONAI integrated)
from dacapo_monai import iterable_dataset
from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd

transforms = Compose([
    ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=1, b_min=0.0, b_max=1.0),
    RandSpatialCropd(keys=["raw"], roi_size=(32, 32, 32))
])

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=transforms  # Full MONAI ecosystem!
)
```

### Using Transform Presets

Skip the boilerplate with expert-crafted presets:

```python
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

# One line for production-ready SSL transforms
ssl_transforms = SSLTransforms.contrastive_3d()

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=ssl_transforms
)
```

## 📚 Examples

### 1. Basic Medical Imaging

```python
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import MedicalImagingTransforms
from funlib.persistence.arrays.array import Array
import numpy as np

# Create sample data
data = np.random.rand(200, 200, 200).astype(np.float32)
array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

# Use medical imaging preset
transforms = MedicalImagingTransforms.basic_3d_preprocessing()

# Create dataset
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=transforms
)

# Use in training
for batch in dataset:
    processed_data = batch['raw']  # Already perfectly preprocessed!
```

### 2. Self-Supervised Learning (SSL)

```python
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

# Get contrastive learning transforms
ssl_transforms = SSLTransforms.contrastive_3d()

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=ssl_transforms
)

# Training loop for contrastive learning
for batch in dataset:
    anchor = batch['raw']      # Original view
    positive = batch['raw_2']   # Transformed view
    
    # Your SSL training logic here
    embeddings_anchor = model(anchor)
    embeddings_positive = model(positive)
    loss = contrastive_loss(embeddings_anchor, embeddings_positive)
```

### 3. Custom MONAI Pipeline

```python
from dacapo_monai import iterable_dataset
from monai.transforms import (
    Compose, ScaleIntensityRanged, RandSpatialCropd, 
    RandRotated, RandGaussianNoised, Lambda
)
import torch

# Build custom MONAI transform pipeline
custom_transforms = Compose([
    # Convert to tensor with channel dimension
    Lambda(func=lambda x: {
        **x, 
        "raw": torch.from_numpy(x["raw"].numpy()).unsqueeze(0).float()
    }),
    
    # MONAI transforms
    ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=1, b_min=0.0, b_max=1.0),
    RandSpatialCropd(keys=["raw"], roi_size=(32, 32, 32)),
    RandRotated(keys=["raw"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
    RandGaussianNoised(keys=["raw"], std=0.01, prob=0.2),
    
    # Convert back for DaCapo compatibility
    Lambda(func=lambda x: {
        **x,
        "raw": x["raw"].squeeze(0)
    }),
])

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=custom_transforms
)
```

## 🎯 Transform Presets

### SSL Transforms

```python
from dacapo_monai.transforms import SSLTransforms

# Contrastive learning (creates two augmented views)
transforms = SSLTransforms.contrastive_3d()

# Available methods:
# - contrastive_3d(): For contrastive learning
# - [More coming soon...]
```

### Medical Imaging Transforms

```python
from dacapo_monai.transforms import MedicalImagingTransforms

# Standard medical preprocessing
transforms = MedicalImagingTransforms.basic_3d_preprocessing()

# Available methods:
# - basic_3d_preprocessing(): Standard normalization and augmentation
# - [More coming soon...]
```

### Contrastive Learning Transforms

```python
from dacapo_monai.transforms import ContrastiveLearningTransforms

# SimCLR-style transforms
transforms = ContrastiveLearningTransforms.simclr_3d()

# BYOL-style transforms  
transforms = ContrastiveLearningTransforms.byol_3d()
```

## 🔧 Advanced Usage

### Custom Transform Adapters

```python
from dacapo_monai.utils import MonaiToDacapoAdapter

# Create adapter for complex MONAI transforms
adapter = MonaiToDacapoAdapter(your_monai_transform)

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=adapter
)
```

### Format Utilities

```python
from dacapo_monai.utils import add_channel_dim, remove_channel_dim

# Manual format conversion
data_with_channels = add_channel_dim(your_data, "raw")
data_without_channels = remove_channel_dim(data_with_channels, "raw")
```

## 🏗️ Architecture

DaCapo-MONAI provides:

1. **Enhanced PipelineDataset**: Automatically detects transform types and applies appropriate handling
2. **Type-Safe Integration**: Full type annotations for better IDE support
3. **Metadata Preservation**: Maintains DaCapo metadata through MONAI transforms  
4. **Flexible Transform System**: Supports both DaCapo dict-style and MONAI callable transforms
5. **Performance Optimized**: Minimal overhead for seamless integration

## 🤝 Migration from DaCapo

Migrating existing code is simple:

```python
# 1. Change import
# from dacapo_toolbox.dataset import iterable_dataset
from dacapo_monai import iterable_dataset

# 2. Add MONAI transforms (optional)
from dacapo_monai.transforms import SSLTransforms
transforms = SSLTransforms.contrastive_3d()

# 3. Update dataset creation
dataset = iterable_dataset(
    datasets=datasets,
    shapes=shapes,
    transforms=transforms  # Can now use MONAI transforms!
)

# 4. Training loop stays the same!
for batch in dataset:
    # Your existing training code works unchanged
    pass
```

## 📖 Documentation

- [API Reference](docs/api.md)
- [Transform Gallery](docs/transforms.md)
- [Migration Guide](docs/migration.md)
- [Examples](examples/)

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific examples
python examples/basic_medical_imaging.py
python examples/ssl_contrastive_learning.py
python examples/migration_example.py
```

## 🤔 Why DaCapo-MONAI?

### Before DaCapo-MONAI

```python
# Separate systems - more complexity
dataset = dacapo_dataset(...)
monai_transforms = Compose([...])

for batch in dataset:
    # Manual format conversion
    batch_dict = convert_format(batch)
    batch_dict = monai_transforms(batch_dict)  
    batch = convert_back(batch_dict)
    # Finally ready for training...
```

### After DaCapo-MONAI

```python
# Unified system - much cleaner  
transforms = SSLTransforms.contrastive_3d()
dataset = iterable_dataset(..., transforms=transforms)

for batch in dataset:
    # Ready to use immediately!
    model_input = batch['raw']
```

## 🔬 Use Cases

- **Self-Supervised Learning**: Contrastive learning on large 3D medical datasets
- **Medical Image Preprocessing**: Standard medical imaging augmentation pipelines  
- **Multi-Modal Learning**: Complex transform pipelines for multiple data modalities
- **Research Prototyping**: Quick experimentation with MONAI's extensive transform library
- **Production Pipelines**: Robust, tested transforms for medical imaging applications

## 🎯 Key Features

- ✅ **Drop-in Replacement**: Works with existing DaCapo workflows
- ✅ **Full MONAI Support**: Access to 100+ medical imaging transforms
- ✅ **Type Safety**: Complete type annotations for better development experience  
- ✅ **Performance**: Minimal overhead, maximum compatibility
- ✅ **Preset Library**: Expert-crafted transform compositions
- ✅ **Flexible**: Mix and match DaCapo and MONAI transforms as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/dacapo-monai/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/dacapo-monai/discussions)
- 📧 **Email**: support@dacapo-monai.org

## 🏆 Acknowledgments

- [DaCapo Toolbox](https://github.com/dacapo-toolbox/dacapo-toolbox) for large-scale 3D dataset processing
- [MONAI](https://github.com/Project-MONAI/MONAI) for medical imaging transforms
- [Gunpowder](https://github.com/funkelab/gunpowder) for pipeline architecture

---

**Made with ❤️ for the medical imaging and self-supervised learning communities**