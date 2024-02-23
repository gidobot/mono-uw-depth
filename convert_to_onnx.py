from os.path import join
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Resize


from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_heatmap

# from datasets.datasets import get_flsea_dataset
from data.example_dataset.dataset import get_example_dataset


BATCH_SIZE = 1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
MODEL_PATH = (
    "data/saved_models/model_e11_udfnet_lr0.0001_bs6_lrd0.9_with_infguidance.pth"
)
DATASET = get_example_dataset(train=False, shuffle=False, device=DEVICE)
OUT_PATH = "data/saved_models"

@torch.no_grad()
def convert():

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
    for batch_id, data in enumerate(dataloader):

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        onnx_model_path = OUT_PATH + "/model.onnx"
        torch.onnx.export(
            model,                  # PyTorch Model
            args=(rgb, prior),      # Input tensor tuple
            f=onnx_model_path,        # Output file (eg. 'output_model.onnx')
            opset_version=14,       # Operator support version
            input_names=['input0', 'input1'],   # Input tensor name (arbitary)
            output_names=['output'], # Output tensor name (arbitary)
            export_params=True
        )

        return


if __name__ == "__main__":
    convert()
