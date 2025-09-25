# Quickstart

Get started with DaCapo-MONAI in 5 minutes!

## Basic Usage

Here's a minimal example showing how to use DaCapo-MONAI:

```python
import numpy as np
import torch
from funlib.persistence.arrays.array import Array
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

# 1. Create sample data
data = np.random.rand(200, 200, 200).astype(np.float32)
array = Array(data, offset=(0, 0, 0), voxel_size=(1, 1, 1))

# 2. Create SSL transforms
transforms = SSLTransforms.contrastive_3d()

# 3. Create dataset with MONAI transforms
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=transforms
)

# 4. Use in training loop
for batch in dataset:
    anchor = batch['raw']      # Original view
    positive = batch['raw_2']   # Augmented view
    
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    
    # Your training code here
    break  # Just show one batch for demo
```

## Transform Presets

DaCapo-MONAI includes pre-configured transform presets for common use cases:

### Self-Supervised Learning

```python
from dacapo_monai.transforms import SSLTransforms

# Contrastive learning transforms
ssl_transforms = SSLTransforms.contrastive_3d()

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=ssl_transforms
)
```

### Medical Imaging

```python
from dacapo_monai.transforms import MedicalImagingTransforms

# Standard medical preprocessing
medical_transforms = MedicalImagingTransforms.basic_3d_preprocessing()

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=medical_transforms
)
```

### Contrastive Learning Methods

```python
from dacapo_monai.transforms import ContrastiveLearningTransforms

# SimCLR-style transforms
simclr_transforms = ContrastiveLearningTransforms.simclr_3d()

# BYOL-style transforms
byol_transforms = ContrastiveLearningTransforms.byol_3d()
```

## Custom MONAI Transforms

You can also use custom MONAI transforms:

```python
from monai.transforms import Compose, ScaleIntensityRanged, RandSpatialCropd
from dacapo_monai.utils import MonaiToDacapoAdapter

# Create custom MONAI transform pipeline
custom_transforms = Compose([
    ScaleIntensityRanged(keys=["raw"], a_min=0, a_max=1, b_min=0.0, b_max=1.0),
    RandSpatialCropd(keys=["raw"], roi_size=(32, 32, 32), random_center=True),
])

# Adapt for DaCapo compatibility
adapted_transforms = MonaiToDacapoAdapter(custom_transforms)

dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=adapted_transforms
)
```

## Complete Training Example

Here's a complete self-supervised learning example:

```python
import torch
import torch.nn as nn
from dacapo_monai import iterable_dataset
from dacapo_monai.transforms import SSLTransforms

# 1. Create dataset with SSL transforms
transforms = SSLTransforms.contrastive_3d()
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=transforms
)

# 2. Define a simple encoder
class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 128)
        )
    
    def forward(self, x):
        return self.conv(x.unsqueeze(1))  # Add channel dim

# 3. Training loop
model = SimpleEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Simplified loss for demo

for epoch in range(2):  # Just 2 epochs for demo
    for i, batch in enumerate(dataset):
        if i >= 5:  # Just 5 batches for demo
            break
            
        # Get contrastive views
        anchor = batch['raw'].unsqueeze(0)     # Add batch dim
        positive = batch['raw_2'].unsqueeze(0) # Add batch dim
        
        # Forward pass
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        
        # Simplified contrastive loss (normally you'd use InfoNCE)
        loss = criterion(anchor_embedding, positive_embedding)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

print("✅ Training completed!")
```

## What's Next?

Now that you've seen the basics, explore more:

- **Migration Guide**: If you have existing DaCapo code, see our [migration guide](migration.md)
- **Examples**: Check out more detailed [examples](../examples/index.md)
- **User Guide**: Learn about all features in the [user guide](../user_guide/index.md)
- **API Reference**: Dive deep into the [API documentation](../api_reference/index.md)

## Common Patterns

### Multi-Modal Data

```python
# Handle multiple data types
dataset = iterable_dataset(
    datasets={
        "raw": raw_array,
        "labels": label_array
    },
    shapes={
        "raw": (64, 64, 64),
        "labels": (64, 64, 64)
    },
    transforms=transforms
)
```

### Custom Shapes

```python
# Different shapes for different data
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (128, 128, 64)},  # Custom shape
    transforms=transforms
)
```

### No Transforms

```python
# Use without transforms (just DaCapo functionality)
dataset = iterable_dataset(
    datasets={"raw": array},
    shapes={"raw": (64, 64, 64)},
    transforms=None  # No transforms
)
```