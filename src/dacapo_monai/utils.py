"""
Utility functions for MONAI-DaCapo integration.

This module provides helper functions for common operations when working
with MONAI transforms in DaCapo pipelines.
"""

from typing import Any, Dict, Union, Optional, List, Callable
import torch
import numpy as np


def add_channel_dim(
    batch: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Add channel dimension to specified keys in a batch.

    MONAI transforms typically expect a channel dimension, while DaCapo
    data often doesn't have one. This function adds the channel dimension.

    Args:
        batch: Dictionary containing the data batch
        keys: Keys to process. If None, processes all tensor keys

    Returns:
        Modified batch with channel dimensions added

    Example:
        batch = {"raw": torch.rand(64, 64, 64)}
        batch = add_channel_dim(batch, ["raw"])
        # batch["raw"].shape is now (1, 64, 64, 64)
    """
    if keys is None:
        keys = [
            k for k, v in batch.items() if isinstance(v, (torch.Tensor, np.ndarray))
        ]

    result = batch.copy()
    for key in keys:
        if key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].unsqueeze(0)
            elif isinstance(result[key], np.ndarray):
                result[key] = torch.from_numpy(result[key]).unsqueeze(0).float()

    return result


def remove_channel_dim(
    batch: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Remove channel dimension from specified keys in a batch.

    After MONAI processing, you may want to remove the channel dimension
    to match DaCapo's expected format.

    Args:
        batch: Dictionary containing the data batch
        keys: Keys to process. If None, processes all tensor keys

    Returns:
        Modified batch with channel dimensions removed

    Example:
        batch = {"raw": torch.rand(1, 64, 64, 64)}
        batch = remove_channel_dim(batch, ["raw"])
        # batch["raw"].shape is now (64, 64, 64)
    """
    if keys is None:
        keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]

    result = batch.copy()
    for key in keys:
        if key in result and isinstance(result[key], torch.Tensor):
            if result[key].dim() > 3:  # Only remove if more than 3D
                result[key] = result[key].squeeze(0)

    return result


def ensure_tensor(
    batch: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Ensure specified keys contain PyTorch tensors.

    Converts numpy arrays to PyTorch tensors while preserving other data types.

    Args:
        batch: Dictionary containing the data batch
        keys: Keys to process. If None, processes all numpy array keys

    Returns:
        Modified batch with tensors

    Example:
        batch = {"raw": np.random.rand(64, 64, 64)}
        batch = ensure_tensor(batch, ["raw"])
        # batch["raw"] is now a torch.Tensor
    """
    if keys is None:
        keys = [k for k, v in batch.items() if isinstance(v, np.ndarray)]

    result = batch.copy()
    for key in keys:
        if key in result and isinstance(result[key], np.ndarray):
            result[key] = torch.from_numpy(result[key]).float()

    return result


def preserve_metadata(
    transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Decorator to preserve metadata when applying transforms.

    This decorator ensures that the 'metadata' key is preserved
    when applying transforms that might not handle it properly.

    Args:
        transform_func: Transform function to wrap

    Returns:
        Wrapped function that preserves metadata

    Example:
        @preserve_metadata
        def my_transform(batch):
            # Transform logic here
            return transformed_batch
    """

    def wrapper(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Extract metadata
        metadata = batch.pop("metadata", None)

        # Apply transform
        result = transform_func(batch)

        # Restore metadata
        if metadata is not None:
            result["metadata"] = metadata

        return result

    return wrapper


def create_channel_wrapper(
    keys: list[str],
) -> Callable[
    [Callable[[Dict[str, Any]], Dict[str, Any]]],
    Callable[[Dict[str, Any]], Dict[str, Any]],
]:
    """
    Create a transform wrapper that handles channel dimensions automatically.

    This creates a function that adds channel dimensions before MONAI transforms
    and removes them afterward.

    Args:
        keys: Keys that need channel dimension handling

    Returns:
        Transform wrapper function

    Example:
        from monai.transforms import ScaleIntensityRanged

        wrapper = create_channel_wrapper(["raw"])
        transform = ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=255, b_min=0.0, b_max=1.0)

        wrapped_transform = lambda batch: remove_channel_dim(
            transform(add_channel_dim(batch, keys)),
            keys
        )
    """

    def wrapper(
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def wrapped_transform(batch: Dict[str, Any]) -> Dict[str, Any]:
            # Add channel dimensions
            batch_with_channels = add_channel_dim(batch, keys)

            # Apply transform
            result = transform_func(batch_with_channels)

            # Remove channel dimensions
            result = remove_channel_dim(result, keys)

            return result

        return wrapped_transform

    return wrapper


def batch_to_monai_format(
    batch: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert a DaCapo batch to MONAI-compatible format.

    This function:
    1. Converts numpy arrays to tensors
    2. Adds channel dimensions where needed
    3. Preserves metadata separately

    Args:
        batch: DaCapo batch dictionary
        keys: Keys to process. If None, processes all array/tensor keys

    Returns:
        MONAI-compatible batch
    """
    if keys is None:
        keys = [
            k
            for k, v in batch.items()
            if isinstance(v, (torch.Tensor, np.ndarray)) and k != "metadata"
        ]

    # Start with the original batch
    result = batch.copy()

    # Ensure tensors
    result = ensure_tensor(result, keys)

    # Add channel dimensions
    result = add_channel_dim(result, keys)

    return result


def batch_from_monai_format(
    batch: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert a MONAI batch back to DaCapo-compatible format.

    This function:
    1. Removes extra channel dimensions
    2. Preserves the metadata structure DaCapo expects

    Args:
        batch: MONAI batch dictionary
        keys: Keys to process. If None, processes all tensor keys

    Returns:
        DaCapo-compatible batch
    """
    if keys is None:
        keys = [
            k
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k != "metadata"
        ]

    # Remove channel dimensions
    result = remove_channel_dim(batch, keys)

    return result


class MonaiToDacapoAdapter:
    """
    A class-based adapter for converting between MONAI and DaCapo formats.

    This adapter can be used to wrap MONAI transforms and handle format
    conversions automatically.

    Example:
        from monai.transforms import Compose, ScaleIntensityRanged

        monai_transforms = Compose([
            ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=255, b_min=0.0, b_max=1.0)
        ])

        adapter = MonaiToDacapoAdapter(monai_transforms, keys=["raw"])

        # Use in DaCapo pipeline
        dataset = iterable_dataset(
            datasets={"raw": my_array},
            shapes={"raw": (128, 128, 128)},
            transforms=adapter
        )
    """

    def __init__(
        self,
        monai_transforms: Callable[[Dict[str, Any]], Dict[str, Any]],
        keys: list[str],
    ) -> None:
        """
        Initialize the adapter.

        Args:
            monai_transforms: MONAI transform or Compose object
            keys: Keys that need format conversion
        """
        self.monai_transforms = monai_transforms
        self.keys = keys

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply MONAI transforms with automatic format conversion.

        Args:
            batch: DaCapo batch dictionary

        Returns:
            Transformed batch in DaCapo format
        """
        # Convert to MONAI format
        monai_batch = batch_to_monai_format(batch, self.keys)

        # Apply MONAI transforms
        transformed_batch = self.monai_transforms(monai_batch)

        # Convert back to DaCapo format
        dacapo_batch = batch_from_monai_format(transformed_batch, self.keys)

        return dacapo_batch


def validate_batch_keys(batch: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that a batch contains required keys.

    Args:
        batch: Batch dictionary to validate
        required_keys: List of required keys

    Raises:
        KeyError: If any required key is missing

    Example:
        validate_batch_keys(batch, ["raw", "gt"])
    """
    missing_keys = [key for key in required_keys if key not in batch]
    if missing_keys:
        raise KeyError(f"Missing required keys in batch: {missing_keys}")


def get_batch_info(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about a batch for debugging.

    Args:
        batch: Batch dictionary

    Returns:
        Dictionary with batch information

    Example:
        info = get_batch_info(batch)
        print(f"Batch keys: {info['keys']}")
        print(f"Tensor shapes: {info['shapes']}")
    """
    info = {
        "keys": list(batch.keys()),
        "shapes": {},
        "dtypes": {},
        "has_metadata": "metadata" in batch,
    }

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            info["shapes"][key] = list(value.shape)
            info["dtypes"][key] = str(value.dtype)
        elif isinstance(value, np.ndarray):
            info["shapes"][key] = list(value.shape)
            info["dtypes"][key] = str(value.dtype)
        else:
            info["dtypes"][key] = type(value).__name__

    return info
