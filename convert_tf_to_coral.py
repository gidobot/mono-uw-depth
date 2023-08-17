import tensorflow as tf
from torch.utils.data import DataLoader
from data.example_dataset.dataset import get_example_dataset
# from datasets.datasets import get_flsea_dataset
import numpy as np

MODEL_PATH = (
    "data/saved_models/tf"
)
OUT_PATH = "data/saved_models/tf/coral_model.tflite"

# DATASET = get_flsea_dataset(train=False, shuffle=False, device='cpu')
DATASET = get_example_dataset(train=False, shuffle=False, device='cpu')
dataloader = DataLoader(DATASET, batch_size=1, drop_last=True)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    for batch_id, data in enumerate(dataloader):
        rgb = np.array(data[0].to('cpu'))  # RGB image
        prior = np.array(data[3].to('cpu'))  # precomputed features and depth values
        # get sample input data as numpy array 
        yield [prior, rgb]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

# Save the model
with open(OUT_PATH, 'wb') as f:
    f.write(tflite_quant_model)