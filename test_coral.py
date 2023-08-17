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
DEVICE='cpu'
DATASET = get_example_dataset(train=False, shuffle=False, device=DEVICE)
OUT_PATH = "data/out"
SAVE = True

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

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

def set_input_tensor(interpreter, input, idx):
  input_details = interpreter.get_input_details()[idx]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = input
  scale, zero_point = input_details['quantization']
  interpreter.set_tensor(tensor_index, np.uint8(input / scale + zero_point))
  # input_tensor[:, :] = np.uint8(input / scale + zero_point)

total_time_per_image = 0.0
n_batches = len(dataloader)
for batch_id, data in enumerate(dataloader):

    # inputs
    rgb = np.array(data[0].to(DEVICE))  # RGB image
    prior = np.array(data[3].to(DEVICE))  # precomputed features and depth values

    # nullprior
    # prior[:, :, :, :] = 0.0

    # import pdb; pdb.set_trace()
    set_input_tensor(interpreter, prior, 0)
    set_input_tensor(interpreter, rgb, 1)

    # outputs
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])
    # Outputs from the TFLite model are uint8, so we dequantize the results:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    # time per img
    time_per_img = (end_time - start_time)
    total_time_per_image += time_per_img

    # heatmap for visuals
    heatmap = gray_to_heatmap(output)  # for visualization

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
