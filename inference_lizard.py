import os
from os.path import join
from os import path as osp
import time
import numpy as np
import cv2

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize


from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_heatmap

# from datasets.datasets import get_flsea_dataset
from data.lizard.dataset import get_lizard_dataset


# BATCH_SIZE = 6
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
# MODEL_PATH = "data/saved_models/model_e11_udfnet_lr0.0001_bs6_lrd0.9_with_infguidance.pth"
MODEL_PATH = "data/saved_models/model_e22_udfnet_lr0.0001_bs6_lrd0.9.pth"
DATASET = get_lizard_dataset(train=False, shuffle=False, device=DEVICE)
OUT_PATH = "data/out/lizard/std"
SAVE = True

if not osp.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

@torch.no_grad()
def inference():

    # device info
    print(f"Using device {DEVICE}")

    # model
    print(f"Loading model from {MODEL_PATH}")
    model = UDFNet(n_bins=80).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loading model done.")

    # dataloader
    dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)

    total_time_per_image = 0.0
    n_batches = len(dataloader)

    dataiter = iter(dataloader)

    # initialize matplotlib video writer
    # Writer = matplotlib.animation.writers['ffmpeg']
    # writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    # Create a VideoWriter object to save the video as mp4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('data/out/lizard/std/lizard.mp4', fourcc, 5, (2*640, 480))

    # fig, axs = plt.subplots(6, 1, figsize=(5, 20))
    fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for batch_id in range(n_batches):
        try:
            data = next(dataiter)
        except Exception as e:
            continue

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        prior = data[3].to(DEVICE)  # precomputed features and depth values
        feats = data[4].to('cpu').squeeze(0)

        # nullprior
        # prior[:, :, :, :] = 0.0

        # outputs
        start_time = time.time()
        prediction, _ = model(rgb, prior)  # prediction in metric scale
        end_time = time.time()

        # time per img
        time_per_img = (end_time - start_time) / rgb.size(0)
        total_time_per_image += time_per_img

        # heatmap for visuals
        heatmap = gray_to_heatmap(prediction).to(DEVICE)  # for visualization

        # save outputs
        if SAVE:
            # resize = Resize(heatmap.size()[-2:])
            resize = Resize(rgb.size()[-2:], antialias=True)
            for i in range(rgb.size(0)):
                index = batch_id * BATCH_SIZE + i

                # out_rgb = join(OUT_PATH, f"{index}_rgb.png")
                # out_prediction = join(OUT_PATH, f"{index}_depth.png")
                out_heatmap = join(OUT_PATH, f"{index}_heatmap.png")
                out_rgb_heatmap = join(OUT_PATH, f"{index}_rgb_heatmap.png")
                out_depth = join(OUT_PATH, f"{index}_depth.png")
                out_fig = join(OUT_PATH, f"{index}.png")

                # save_image(rgb[i], out_rgb)
                # save_image(prediction[i], out_prediction)
                # save_image(heatmap[i], out_heatmap)
                # save_image([rgb[i], resize(heatmap[i])], out_rgb_heatmap)

                # save_image(prediction[i], out_depth)

            colormap = plt.get_cmap("inferno_r")

            # pred = prediction[0].permute(1,2,0).to('cpu')
            pred = heatmap[0].permute(1,2,0).to('cpu')
            d_img = data[1].to('cpu')[0, ...] # depth image
            d_img = d_img.permute(1,2,0)
            prior_feats = prior[0,0,:,:].unsqueeze(0).permute(1,2,0).to('cpu')
            prior_depth = prior[0,1,:,:].unsqueeze(0).permute(1,2,0).to('cpu')
            error = np.abs(d_img-pred)

            rgb = rgb[0].permute(1,2,0).to('cpu')

            # draw points on image
            # plt.scatter(feats[:,0]*pred.size(1), feats[:,1]*pred.size(0), s=4, c='b')
            # for i in range(feats.size(0)):
            #     r = feats[i, 0]*pred.size(0)
            #     c = feats[i, 1]*pred.size(1)
            #     plt.scatter(c, r, s=4, c='b')
            #     r = feats[i, 0]*rgb.size(0)
            #     c = feats[i, 1]*rgb.size(1)
            #     plt.scatter(c, r, s=4, c='b')

            axs[0].clear()
            axs[0].imshow(rgb)
            axs[0].scatter(feats[:,1]*2, feats[:,0]*2, s=4, c='b')
            axs[0].axis('off')
            # axs[1].imshow(d_img)
            axs[1].clear()
            axs[1].imshow(pred)
            axs[1].scatter(feats[:,1], feats[:,0], s=4, c='b')
            axs[1].axis('off')
            # axs[2].imshow(prior_depth)
            # axs[2].axis('off')
            # axs[3].imshow(prior_feats)
            # axs[3].axis('off')
            # axs[4].imshow(pred)
            # axs[4].axis('off')
            # axs[5].imshow(error)
            # axs[5].axis('off')
            plt.savefig(out_fig)
            # plt.close()
            # plt.show()

            frame = cv2.imread(out_fig)
            out.write(frame)

        if batch_id % 10 == 0:
            print(f"{batch_id}/{n_batches}, {1.0/time_per_img} FPS")

    # Release the VideoWriter object
    out.release()

    avg_time_per_image = total_time_per_image / n_batches
    avg_fps = 1.0 / avg_time_per_image

    print(f"Average time per image: {avg_time_per_image}")
    print(f"Average FPS: {avg_fps}")


if __name__ == "__main__":
    inference()
