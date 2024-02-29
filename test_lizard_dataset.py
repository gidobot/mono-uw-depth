import os 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

# from datasets.datasets import get_flsea_dataset
from data.lizard.dataset import get_lizard_dataset
from depth_estimation.utils.visualization import gray_to_heatmap


BATCH_SIZE = 1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
DATASET = get_lizard_dataset(train=False, shuffle=False, device=DEVICE)

dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)

# for batch_id, data in enumerate(dataloader):

#     rgb_imgs = data[0]
#     d_imgs = data[1]
#     masks = data[2]
#     parametrizations = data[3]

#     for i in range(rgb_imgs.size(0)):

#         rgb_img = rgb_imgs[i, ...]
#         d_img = d_imgs[i, ...]
#         mask = masks[i, ...]
#         nn_parametrization = parametrizations[i, 0, ...].unsqueeze(0)
#         prob_parametrization = parametrizations[i, 1, ...].unsqueeze(0)

#         print(f"d range: [{d_img.min()}, {d_img.max()}]")

#         plt.figure(f"rgb img {i}")
#         plt.imshow(rgb_img.permute(1, 2, 0))
#         plt.figure(f"d img {i}")
#         plt.imshow(d_img.permute(1, 2, 0))
#         plt.figure(f"mask {i}")
#         plt.imshow(mask.permute(1, 2, 0))
#         plt.figure(f"parametrization, NN {i}")
#         plt.imshow(nn_parametrization.permute(1, 2, 0))
#         plt.figure(f"parametrization, Probability {i}")
#         plt.imshow(prob_parametrization.permute(1, 2, 0))

#     plt.show()

#     break  # only check first batch


for batch_id, data in enumerate(dataloader):
    # inputs
    rgb = data[0].to(DEVICE)[0, ...]  # RGB image
    d_img = data[1].to(DEVICE)[0, ...] # depth image
    prior = data[3].to(DEVICE)  # precomputed features and depth values

    feat_depth = data[4].to(DEVICE).squeeze()

    prior_depth = prior[0,0,:,:].unsqueeze(0).permute(1,2,0)
    prior_feats = prior[0,1,:,:].unsqueeze(0).permute(1,2,0)
    # heat_feats = gray_to_heatmap(prior_feats).to(DEVICE)  # for visualization
    # heat_feats = heat_feats.squeeze()
    # heat_depth = gray_to_heatmap(prior_depth).to(DEVICE)  # for visualization
    # heat_depth = heat_depth.squeeze()
    rgb = rgb.permute(1,2,0)
    d_img = d_img.permute(1,2,0)

    se = 0
    for i in range(feat_depth.shape[1]):
        r = int(feat_depth[i,0])
        c = int(feat_depth[i,1])
        df = feat_depth[i,2]
        dd = d_img[r,c]
        se = se + df/dd
    se = se/feat_depth.shape[1]
    
    print("Mean scale error: {}".format(se))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0,0].imshow(prior_feats)
    axs[0,1].imshow(prior_depth)
    axs[1,0].imshow(rgb)
    axs[1,1].imshow(d_img)
    plt.show()

print("Testing DataSet class done.")