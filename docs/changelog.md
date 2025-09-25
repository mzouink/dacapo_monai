# Changelog

All notable changes to DaCapo-MONAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation structure
- GitHub Actions for CI/CD
- Sphinx documentation with Furo theme

### Changed
- Enhanced pyproject.toml with documentation dependencies

### Fixed
- Documentation build configuration

## [1.0.0] - 2025-01-25

### Added
- 🎉 **Initial Release** of DaCapo-MONAI
- **Core Integration**: Seamless MONAI transforms with DaCapo datasets
- **Transform Presets**: Expert-crafted presets for SSL, medical imaging, and contrastive learning
- **Type Safety**: Complete type annotations with Union types for backward compatibility
- **Format Utilities**: Automatic conversion between DaCapo and MONAI data formats
- **Professional Packaging**: Modern pyproject.toml configuration ready for PyPI
- **Comprehensive Examples**: Real-world usage examples for medical imaging and SSL
- **Test Suite**: Complete testing framework with pytest
- **Documentation Structure**: Sphinx-based documentation framework

### Core Features
- ✅ **Enhanced PipelineDataset**: Supports both DaCapo dict transforms and MONAI callable transforms
- ✅ **Automatic Transform Detection**: Intelligently handles different transform types
- ✅ **Metadata Preservation**: Maintains DaCapo metadata through MONAI transform pipelines
- ✅ **Transform Presets**:
  - `SSLTransforms.contrastive_3d()`: Contrastive learning with dual views
  - `MedicalImagingTransforms.basic_3d_preprocessing()`: Standard medical preprocessing
  - `ContrastiveLearningTransforms.simclr_3d()`: SimCLR-style transforms
  - `ContrastiveLearningTransforms.byol_3d()`: BYOL-style transforms
- ✅ **Utility Functions**:
  - `add_channel_dim()`: Add MONAI-compatible channel dimensions
  - `remove_channel_dim()`: Remove channels for DaCapo compatibility
  - `MonaiToDacapoAdapter`: Automatic adapter for MONAI transforms

### Examples Included
- **Basic Medical Imaging**: Standard preprocessing workflow
- **SSL Contrastive Learning**: Self-supervised learning setup
- **Migration Guide**: Complete before/after code examples
- **End-to-End Demo**: Full functionality demonstration

### Technical Achievements
- **Type System Enhancement**: Enhanced DaCapo's transform typing from simple dict to Union types
- **Backward Compatibility**: Maintains full compatibility with existing DaCapo workflows
- **Performance Optimized**: Minimal overhead for seamless integration
- **Developer Experience**: Complete type annotations and comprehensive documentation

### Dependencies
- `torch >= 1.12.0`
- `monai >= 1.0.0`
- `dacapo-toolbox >= 1.0.0`
- `gunpowder >= 1.4.0`
- `funlib.geometry >= 0.2.0`
- `funlib.persistence >= 0.6.0`
- `numpy >= 1.21.0`

### Installation
```bash
pip install dacapo-monai
```

### Migration from DaCapo
Simple migration path:
```python
# Before
from dacapo_toolbox.dataset import iterable_dataset
# After  
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms
transforms = SSLTransforms.contrastive_3d()
```

---

**Legend**:
- 🎉 Major milestone
- ✅ Feature complete
- 🔧 Enhancement
- 🐛 Bug fix
- ⚠️ Breaking change
- 📚 Documentation
- 🚀 Performance improvement