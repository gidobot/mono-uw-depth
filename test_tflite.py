from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from depth_estimation.model.model import UDFNet

# from datasets.datasets import get_flsea_dataset
from data.example_dataset.dataset import get_example_dataset

import tensorflow as tf

BATCH_SIZE = 1
MODEL_PATH = ("data/saved_models/tf/coral_model.tflite")
# MODEL_PATH = ("data/saved_models/tf/model.tflite")
DEVICE='cpu'
DATASET = get_example_dataset(train=False, shuffle=False, device=DEVICE)
OUT_PATH = "data/out"
SAVE = True

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# dataloader
dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)

def gray_to_heatmap(gray, colormap="inferno_r", normalize=True):
    """Takes torch tensor input of shape [Nx1HxW], returns heatmap tensor of shape [Nx3xHxW].\\
    colormap 'inferno_r': [0,1] --> [bright, dark], e.g. for depths\\
    colormap 'inferno': [0,1] --> [dark, bright], e.g. for probabilities"""

    # get colormap
    colormap = plt.get_cmap(colormap)

    # gray imgs
    gray_img = gray

    # normalize image wise
    if normalize:
        gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())

    # stack heatmaps batch wise (colormap does not support batches)
    heatmap = colormap(gray_img.squeeze())[..., :3]

    return heatmap

total_time_per_image = 0.0
n_batches = len(dataloader)
for batch_id, data in enumerate(dataloader):

    # inputs
    rgb = np.array(data[0].to(DEVICE))  # RGB image
    prior = np.array(data[3].to(DEVICE))  # precomputed features and depth values

    # nullprior
    # prior[:, :, :, :] = 0.0

    # import pdb; pdb.set_trace()
    interpreter.set_tensor(input_details[0]['index'], prior)
    interpreter.set_tensor(input_details[1]['index'], rgb)

    # outputs
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # time per img
    time_per_img = (end_time - start_time)
    total_time_per_image += time_per_img

    # heatmap for visuals
    heatmap = gray_to_heatmap(output_data)  # for visualization

    # save outputs
    if SAVE:
        resize = Resize(rgb.shape[-2:])
        index = batch_id * BATCH_SIZE
        out_heatmap = join(OUT_PATH, f"{index}_heatmap.png")
        out_rgb_heatmap = join(OUT_PATH, f"{index}_rgb_heatmap.png")
        rgb = rgb*255. + 0.5
        rgb = rgb.clip(0,255)
        heatmap = heatmap*255. + 0.5
        heatmap = heatmap.clip(0,255)
        cv2.imwrite(out_heatmap, heatmap)
        cv2.imwrite(out_rgb_heatmap, rgb.squeeze().transpose((1,2,0)))

    if batch_id % 10 == 0:
        print(f"{batch_id}/{n_batches}, {1.0/time_per_img} FPS")

avg_time_per_image = total_time_per_image / n_batches
avg_fps = 1.0 / avg_time_per_image

print(f"Average time per image: {avg_time_per_image}")
print(f"Average FPS: {avg_fps}")
