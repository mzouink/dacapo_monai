# Utils Module

Format conversion utilities and adapters.

## Overview

The utils module provides helper functions and adapters for seamless integration between MONAI and DaCapo data formats.

## Key Utilities

### Data Format Conversion
- `add_channel_dim`: Add channel dimensions to arrays
- `remove_channel_dim`: Remove channel dimensions from arrays
- `ensure_tensor`: Convert arrays to tensors with proper formatting

### Metadata Management
- `preserve_metadata`: Preserve DaCapo metadata through MONAI transforms
- Format-specific adapters for different data types

### Integration Helpers
- `MonaiToDacapoAdapter`: Main adapter class for format conversion
- Utility functions for common conversion patterns

## API Reference

```{eval-rst}
.. automodule:: dacapo_monai.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :ignore-module-all:
```