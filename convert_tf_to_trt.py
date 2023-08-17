import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from torch.utils.data import DataLoader
from data.example_dataset.dataset import get_example_dataset
# from datasets.datasets import get_flsea_dataset
import numpy as np

MODEL_PATH = ("data/saved_models/tf")
OUT_PATH = "data/saved_models/trt"

# DATASET = get_flsea_dataset(train=False, shuffle=False, device='cpu')
DATASET = get_example_dataset(train=False, shuffle=False, device='cpu')
dataloader = DataLoader(DATASET, batch_size=1, drop_last=True)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=MODEL_PATH,
   precision_mode=trt.TrtPrecisionMode.FP16,
   maximum_cached_engines=100,
   use_dynamic_shape=True)
   # use_calibration=True)

# Use data from the test/validation set to perform INT8 calibration
def representative_dataset_gen():
    for batch_id, data in enumerate(dataloader):
        rgb = np.array(data[0].to('cpu'))  # RGB image
        prior = np.array(data[3].to('cpu'))  # precomputed features and depth values
        # get sample input data as numpy array 
        yield [rgb, prior]


# Convert the model with valid calibration data
# func = converter.convert(calibration_input_fn=representative_dataset_gen)
func = converter.convert()

# Input for dynamic shapes profile generation
dataloader = DataLoader(DATASET, batch_size=1, drop_last=True)
def input_fn():
    for batch_id, data in enumerate(dataloader):
        rgb = np.array(data[0].to('cpu'))  # RGB image
        prior = np.array(data[3].to('cpu'))  # precomputed features and depth values
        # get sample input data as numpy array 
        yield [rgb, prior]
 
# Build the engine
# converter.build(input_fn=input_fn)
# converter.build(input_fn=input_fn)
converter.summary()

converter.save(output_saved_model_dir=OUT_PATH)


# BATCH_SIZE=32
# NUM_CALIB_BATCHES=10
# def calibration_input_fn():
   # for i in range(NUM_CALIB_BATCHES):
       # start_idx = i * BATCH_SIZE
       # end_idx = (i + 1) * BATCH_SIZE
       # x = x_test[start_idx:end_idx, :]
       # yield [x]
 
 
# Run some inferences!
# for step in range(10):
#    start_idx = step * BATCH_SIZE
#    end_idx   = (step + 1) * BATCH_SIZE
 
#    print(f"Step: {step}")
#    x = x_test[start_idx:end_idx, :]
#    func(x)