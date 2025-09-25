# Developer Guide

Contributing to DaCapo-MONAI development.

```{toctree}
:maxdepth: 2

contributing
development_setup
testing
documentation
release_process
```

## Overview

Welcome to the DaCapo-MONAI developer documentation! This guide covers everything you need to know about contributing to the project.

::::{grid} 2
:::{grid-item-card} 🤝 Contributing
:link: contributing
:link-type: doc

Guidelines for contributing code, documentation, and examples.
:::

:::{grid-item-card} 🔧 Development Setup
:link: development_setup
:link-type: doc

Set up your development environment for DaCapo-MONAI.
:::

:::{grid-item-card} 🧪 Testing
:link: testing
:link-type: doc

Testing guidelines and best practices.
:::

:::{grid-item-card} 📚 Documentation
:link: documentation
:link-type: doc

How to build and contribute to documentation.
:::
::::

## Quick Start for Developers

```bash
# Clone the repository
git clone https://github.com/dacapo-toolbox/dacapo-monai.git
cd dacapo-monai

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Build documentation
cd docs
sphinx-build -b html . _build/html
```

## Development Workflow

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Develop**: Make your changes following our coding standards
4. **Test**: Run tests and ensure they pass
5. **Document**: Update documentation as needed
6. **Submit PR**: Submit a pull request with clear description

## Code Standards

- **Type Hints**: All public functions must have type annotations
- **Documentation**: Use Google-style docstrings
- **Testing**: Maintain >90% test coverage
- **Formatting**: Use Black for code formatting
- **Linting**: Code must pass flake8 checks

## Getting Help

- 💬 **Discord**: Join our developer Discord server
- 🐛 **Issues**: Report bugs on GitHub Issues
- 📧 **Email**: Contact maintainers directly
- 📖 **Docs**: Check the documentation first

We're excited to have you contribute! 🚀