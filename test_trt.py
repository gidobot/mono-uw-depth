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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


USE_FP16 = False
target_dtype = np.float16 if USE_FP16 else np.float32

BATCH_SIZE = 1
MODEL_PATH = "data/saved_models/model.engine"
DEVICE='cpu'
DATASET = get_example_dataset(train=False, shuffle=False, device=DEVICE)
OUT_PATH = "data/out"
SAVE = True

f = open(MODEL_PATH, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()


# dataloader
dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)

init_flag = False

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
    rgb = data[0].to('cuda:0')  # RGB image
    prior = data[3].to('cuda:0')  # precomputed features and depth values


    if not init_flag:
        # Use device pointers already in GPU memory for bindings
        outputs = [torch.zeros((1, 1, 240, 320), dtype=torch.float32, device="cuda:0"), torch.zeros((1, 81), dtype=torch.float32, device="cuda:0")]
        # # need to set input and output precisions to FP16 to fully enable it
        # output = np.empty([1, 1, 240, 320], dtype = target_dtype) 
        # # allocate device memory
        # d_input0 = cuda.mem_alloc(1 * rgb.nbytes)
        # d_input1 = cuda.mem_alloc(1 * prior.nbytes)
        # d_output = cuda.mem_alloc(1 * output.nbytes)
        # bindings = [int(d_input0), int(d_input1), int(d_output)]
        # # stream = cuda.Stream()
        init_flag = True
    bindings = [rgb.data_ptr(), prior.data_ptr(), outputs[1].data_ptr(), outputs[0].data_ptr()]

    # outputs
    start_time = time.time()
    # transfer input data to device
    # cuda.memcpy_htod(d_input0, rgb)
    # cuda.memcpy_htod(d_input1, prior)
    # execute model
    # context.execute_async_v2(bindings, stream.handle, None)
    context.execute_v2(bindings)
    # transfer predictions back
    # cuda.memcpy_dtoh(output, d_output)
    # syncronize threads
    # stream.synchronize()
    end_time = time.time()
    # time per img
    time_per_img = (end_time - start_time)
    total_time_per_image += time_per_img

    # heatmap for visuals
    heatmap = gray_to_heatmap(outputs[0].cpu().numpy().copy())  # for visualization

    # save outputs
    if SAVE:
        rgb_loc = rgb[0].cpu().numpy().copy()
        resize = Resize(rgb_loc.shape[-2:])
        index = batch_id * BATCH_SIZE
        out_heatmap = join(OUT_PATH, f"{index}_heatmap.png")
        out_rgb_heatmap = join(OUT_PATH, f"{index}_rgb_heatmap.png")
        rgb_loc = rgb_loc*255. + 0.5
        rgb_loc = rgb_loc.clip(0,255)
        heatmap = heatmap*255. + 0.5
        heatmap = heatmap.clip(0,255)
        cv2.imwrite(out_heatmap, heatmap)
        cv2.imwrite(out_rgb_heatmap, rgb_loc.squeeze().transpose((1,2,0)))

    if batch_id % 10 == 0:
        print(f"{batch_id}/{n_batches}, {1.0/time_per_img} FPS")

avg_time_per_image = total_time_per_image / n_batches
avg_fps = 1.0 / avg_time_per_image

print(f"Average time per image: {avg_time_per_image}")
print(f"Average FPS: {avg_fps}")
