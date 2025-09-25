# %%
import multiprocessing as mp

mp.set_start_method("fork", force=True)
import dask

dask.config.set(scheduler="single-threaded")

from pathlib import Path
from functools import partial
from tqdm import tqdm

from funlib.persistence import Array, open_ds
from funlib.geometry import Coordinate, Roi
from IPython.display import Image
from dacapo_toolbox.vis.preview import gif_2d, cube

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
from dacapo_toolbox.dataset import (
    iterable_dataset,
    DeformAugmentConfig,
    SimpleAugmentConfig,
)

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
