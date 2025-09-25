#%%
from funlib.persistence import open_ds, Array, open_ome_ds


gt_path = "/nrs/cellmap/data/jrc_c-elegans-P3_E5_D1_N2/jrc_c-elegans-P3_E5_D1_N2.zarr/recon-1/labels/inference/segmentations/nuc/s0"
raw_path = "/groups/funceworm/funceworm/adult/Adult_Day1_DatasetA4/jrc_P3_E5_D1_N2_trimmed_align_v2.zarr/s2"
# %%
gt_ds = open_ds(gt_path)
raw_ds = open_ds(raw_path)
# %%

raw_ds = Array(raw_ds.data, offset=(0, 0, 0), voxel_size=(1, 1, 1), axis_names=["z", "y", "x"])
gt_ds = Array(gt_ds.data, offset=(0,0,0), voxel_size=(1, 1, 1), axis_names=["z", "y", "x"])
#%%
print(raw_ds.shape)
print(gt_ds.shape)
#%%
from dacapo_toolbox.dataset import (
    iterable_dataset,
    SimpleAugmentConfig,
    DeformAugmentConfig,
    MaskedSampling,
)
from dacapo_toolbox.transforms.lsds import LSD
from funlib.persistence import Array
from skimage import data
from torchvision.transforms import v2 as transforms
import logging
import numpy as np
from skimage.measure import label

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("gunpowder.nodes.random_location").setLevel(logging.DEBUG)
from dacapo_toolbox.transforms.torch_lsds import get_local_shape_descriptors_torch
#%%

# defining the datasets
iter_ds = iterable_dataset(
    {
        "raw_s0": [raw_ds],
        "gt_s0": [gt_ds],
    },
    shapes={
        "raw_s0": (1,100 , 100 ),  
        # "gt_s0": (1, 128 , 128 ),
        "gt_s0": ( 1,50 , 50 ),
    },
    transforms={

        ("raw_s0", "noisy_s0"): transforms.Compose(
            [transforms.Lambda(lambda x: x[0]), transforms.ConvertImageDtype(), transforms.GaussianNoise(sigma=.1)]
        ),
        ("gt_s0", "lsd_s0"): transforms.Compose(
            [transforms.Lambda(lambda x: get_local_shape_descriptors_torch(x[0], sigma=(10.0, 10.0), voxel_size=(1.0, 1.0), downsample=2))]
        )
        
    },
    sampling_strategies=MaskedSampling("gt_s0",min_masked=0.0,strategy="reject")
    # transforms={
    #     

    # },
    # # sample_points=[np.array([(0, 0)]), np.array([(512, 512), (0, 512), (512, 0)])],
    # simple_augment_config=SimpleAugmentConfig(
    #     p=1.0, mirror_probs=[1.0, 0.0], transpose_only=[]
    # ),
    # deform_augment_config=DeformAugmentConfig(
    #     p=1.0,
    #     control_point_spacing=(1,10, 10),
    #     jitter_sigma=(1,5.0, 5.0),
    #     scale_interval=(1,0.5, 2.0),
    #     rotate=True,
    # ),
)
#%%
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



# Create a DataLoader
dataloader = DataLoader(iter_ds, batch_size=1)
data_iterator = iter(dataloader)
#%%
batch = next(data_iterator)
#%%
batch["lsd_s0"].shape
#%%
while True:
    batch = next(data_iterator)
    gt_batch = batch["gt_s0"][0,0]
    raw_batch = batch["lsd_s0"][0,0]
    if gt_batch.any():
        full_gt = np.zeros((100,100))
        full_gt[25:75, 25:75] = batch["gt_s0"][0,0]



        plt.imshow(batch["raw_s0"][0,0], cmap="gray")
        plt.imshow(full_gt, alpha=0.5, cmap="jet")

        plt.title("Raw")
        plt.title("Ground Truth")

        plt.show()
        break
    else:
        print("empty batch")

# %%
batch["raw_s0"].shape
# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 7, figsize=(15, 8))
axs[ 0].imshow(batch["raw_s0"][0][0], cmap="gray")
axs[ 1].imshow(batch["noisy_s0"][0], cmap="gray")
axs[ 2].imshow(batch["gt_s0"][0][0], cmap="jet")
axs[ 3].imshow(batch["lsd_s0"][0][[0, 1, 5]].permute(1, 2, 0).float())
axs[ 4].imshow(batch["lsd_s0"][0][[2, 3, 5]].permute(1, 2, 0).float())
axs[ 5].imshow(batch["lsd_s0"][0,4], cmap="gray")
axs[ 6].imshow(batch["lsd_s0"][0,5], cmap="gray")

axs[ 0].set_title("Raw")
axs[ 1].set_title("LSD (Offsets)")
axs[ 2].set_title("LSD (Variance)")
axs[ 3].set_title("LSD (Pearson)")
axs[ 4].set_title("LSD (Mass)")

plt.show()

#%%