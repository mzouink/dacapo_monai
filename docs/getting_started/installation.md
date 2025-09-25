# Installation

This guide covers different ways to install DaCapo-MONAI.

## Standard Installation

Install the latest stable version from PyPI:

```bash
pip install dacapo-monai
```

## Development Installation

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/dacapo-toolbox/dacapo-monai.git

# Install in development mode
pip install -e ".[dev]"
```

## Dependencies

DaCapo-MONAI requires the following core dependencies:

- `torch >= 1.10.0`
- `monai >= 1.0.0` 
- `dacapo-toolbox >= 0.1.0`
- `gunpowder >= 1.3.0`
- `funlib.geometry >= 0.1.0`
- `funlib.persistence >= 1.0.0`
- `numpy >= 1.21.0`

### Optional Dependencies

For additional features, you can install optional dependencies:

```bash
# For development and testing
pip install "dacapo-monai[dev]"

# For documentation building
pip install "dacapo-monai[docs]"

# All optional dependencies
pip install "dacapo-monai[all]"
```

## Verification

Verify your installation by running:

```python
import dacapo_monai
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

print(f"DaCapo-MONAI version: {dacapo_monai.__version__}")
print("✅ Installation successful!")
```

## Docker Installation

For containerized environments:

```dockerfile
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install DaCapo-MONAI
RUN pip install dacapo-monai

# Verify installation
RUN python -c "import dacapo_monai; print('✅ DaCapo-MONAI installed')"
```

## Common Issues

### Import Errors

If you encounter import errors:

```bash
# Make sure all dependencies are installed
pip install torch monai dacapo-toolbox gunpowder

# For development installations, ensure the package is in your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/dacapo-monai/src"
```

### CUDA Issues

For CUDA-related issues:

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install DaCapo-MONAI
pip install dacapo-monai
```

### Version Conflicts

If you have version conflicts:

```bash
# Create a fresh environment
conda create -n dacapo-monai python=3.9
conda activate dacapo-monai

# Install DaCapo-MONAI
pip install dacapo-monai
```

## Next Steps

Once installed, head to the [quickstart guide](quickstart.md) to start using DaCapo-MONAI!