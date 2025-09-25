# %%
import multiprocessing as mp
import json
import time
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from tqdm import tqdm

from funlib.persistence import Array, open_ds
from funlib.geometry import Coordinate, Roi
from IPython.display import Image
from dacapo_toolbox.vis.preview import gif_2d, cube
from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)
from dacapo_toolbox.dataset import (
    iterable_dataset,
    DeformAugmentConfig,
    SimpleAugmentConfig,
)

print_config()
mp.set_start_method("fork", force=True)
import dask

dask.config.set(scheduler="single-threaded")


current_path = Path("imgs")
if not (current_path / "_static" / "ssl").exists():
    (current_path / "_static" / "ssl").mkdir(parents=True, exist_ok=True)

data_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s0/"
raw_train = open_ds(data_path)
# %%
print(raw_train.shape, raw_train.voxel_size)
# %%
blocksize = Coordinate(256, 256, 256)

# %%


# %%
train_dataset = iterable_dataset(
    datasets={"raw": raw_train},
    shapes={"raw": blocksize},
    # deform_augment_config=DeformAugmentConfig(
    #     p=0.1,
    #     control_point_spacing=(2, 10, 10),
    #     jitter_sigma=(0.5, 2, 2),
    #     rotate=True,
    #     subsample=4,
    #     rotation_axes=(1, 2),
    #     scale_interval=(1.0, 1.0),
    # ),
    # simple_augment_config=SimpleAugmentConfig(
    #     p=1.0,
    #     mirror_only=(1, 2),
    #     transpose_only=(1, 2),
    # ),
    # trim=Coordinate(5, 5, 5),
)
batch_gen = iter(train_dataset)
# %%
batch = next(batch_gen)
simple_batch_2d_path = current_path / "_static" / "ssl" / "simple-batch.gif"

print("Creating simple batch gif and jpg...")
gif_2d(
    arrays={
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
    },
    array_types={"Raw": "raw"},
    filename=str(simple_batch_2d_path),
    title="Simple Batch",
    fps=10,
)
Image(filename=str(simple_batch_2d_path))
# %%
train_transforms = Compose(
    [
        Spacingd(keys=["raw"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["raw"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["raw"], source_key="raw", allow_smaller=True),
        SpatialPadd(keys=["raw"], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(
            keys=["raw"], roi_size=(96, 96, 96), random_size=False, num_samples=2
        ),
        CopyItemsd(
            keys=["raw"],
            times=2,
            names=["gt_image", "image_2"],
            allow_missing_keys=False,
        ),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["raw"],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=["raw"],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(keys=["raw"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=["image_2"],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)

# %%
train_dataset = iterable_dataset(
    datasets={"raw": raw_train},
    shapes={"raw": blocksize},
    transforms={"raw": train_transforms},
    # deform_augment_config=DeformAugmentConfig(
    #     p=0.1,
    #     control_point_spacing=(2, 10, 10),
    #     jitter_sigma=(0.5, 2, 2),
    #     rotate=True,
    #     subsample=4,
    #     rotation_axes=(1, 2),
    #     scale_interval=(1.0, 1.0),
    # ),
    # simple_augment_config=SimpleAugmentConfig(
    #     p=1.0,
    #     mirror_only=(1, 2),
    #     transpose_only=(1, 2),
    # ),
    # trim=Coordinate(5, 5, 5),
)
batch_gen = iter(train_dataset)
# %%
batch = next(batch_gen)
# %%
simple_batch_2d_path = current_path / "_static" / "ssl" / "simple-batch.gif"

print("Creating simple batch gif and jpg...")
gif_2d(
    arrays={
        "Raw": Array(batch["raw"].numpy(), voxel_size=raw_train.voxel_size),
    },
    array_types={"Raw": "raw"},
    filename=str(simple_batch_2d_path),
    title="Simple Batch",
    fps=10,
)
Image(filename=str(simple_batch_2d_path))

# %%
