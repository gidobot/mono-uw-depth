from os.path import join
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize


from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_heatmap
from depth_estimation.utils.loss import L2Loss
from datasets.datasets import get_flsea_dataset


BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
MODEL_PATH = "/home/auv/depth_estimation/depth_estimation/train_runs_udfnet/experiments/benchmark2/saved_models/model_e22_udfnet_lr0.0001_bs6_lrd0.9.pth"
DATASET = get_flsea_dataset(
    device=DEVICE,
    split="test_with_matched_features",
    train=False,
    use_csv_samples=True,
    shuffle=False,
)
OUT_PATH = "/home/auv/depth_estimation/depth_estimation/out"
SAVE = False

# clamping
# MIN_DEPTH_VISUALIZE = 0.0  # everything beyond is set to max
# MAX_DEPTH_VISUALIZE = 10.0  # everything beyond is set to max

# l2_loss = L2Loss()


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
    # ranges = [1, 2, 3, 4, 5]
    # range_rmse_losses = [[], [], [], [], []]
    # maxd = []
    for batch_id, data in enumerate(dataloader):

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        # target = data[1].to(DEVICE)  # depth image
        # mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        # # nullprior
        # prior[:, :, :, :] = 0.0

        # outputs
        start_time = time.time()
        prediction, _ = model(rgb, prior)
        end_time = time.time()

        # loss
        # for r, loss in zip(ranges, range_rmse_losses):
        # for i in range(len(ranges)):
        #     range_mask = target[mask] < ranges[i]
        #     if range_mask.any():
        #         range_rmse = l2_loss(
        #             prediction[mask][range_mask], target[mask][range_mask]
        #         ).item()
        #         range_rmse_losses[i].append(range_rmse)
        # maxd.append(target[mask].max().item())

        # time per img
        time_per_img = (end_time - start_time) / rgb.size(0)
        total_time_per_image += time_per_img

        # clamp
        # close_mask = prediction < MIN_DEPTH_VISUALIZE
        # far_away_mask = prediction > MAX_DEPTH_VISUALIZE
        # if not close_mask.any():
        #     prediction[:, :, -1, 0] = MIN_DEPTH_VISUALIZE  # make sure min is present
        # else:
        #     prediction[close_mask] = MIN_DEPTH_VISUALIZE
        # if not far_away_mask.any():
        #     prediction[:, :, 0, 0] = MAX_DEPTH_VISUALIZE  # make sure max is present
        # else:
        #     prediction[far_away_mask] = MAX_DEPTH_VISUALIZE

        # heatmap for visuals
        heatmap = gray_to_heatmap(prediction).to(DEVICE)

        # save
        if SAVE:
            # resize = Resize(heatmap.size()[-2:])
            resize = Resize(rgb.size()[-2:])
            for i in range(rgb.size(0)):
                index = batch_id * BATCH_SIZE + i
                # out_rgb = join(OUT_PATH, f"{index}_rgb.png")
                # out_prediction = join(OUT_PATH, f"{index}_depth.png")
                # out_heatmap = join(OUT_PATH, f"{index}_heatmap.png")
                out_rgb_heatmap = join(OUT_PATH, f"{index}_rgb_heatmap.png")
                # save_image(rgb[i], out_rgb)
                # save_image(prediction[i], out_prediction)
                # save_image(heatmap[i], out_heatmap)
                # save_image([resize(rgb[i]), heatmap[i]], out_rgb_heatmap)
                save_image([rgb[i], resize(heatmap[i])], out_rgb_heatmap)

        if batch_id % 10 == 0:
            print(f"{batch_id}/{n_batches}, {1.0/time_per_img} FPS")

    avg_time_per_image = total_time_per_image / n_batches
    avg_fps = 1.0 / avg_time_per_image

    print(f"Average time per image: {avg_time_per_image}")
    print(f"Average FPS: {avg_fps}")

    # for i in range(len(ranges)):
    #     print(f"RMSE[d<{ranges[i]}] mean: {np.mean(range_rmse_losses[i])}]")
    # for i in range(len(ranges)):
    #     print(f"RMSE[d<{ranges[i]}] median: {np.median(range_rmse_losses[i])}]")

    # print(f"maxd mean: {np.mean(maxd)}")
    # print(f"maxd  median: {np.median(maxd)}")


if __name__ == "__main__":
    inference()
