from os.path import join
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

# import onnx
# from onnx_tf.backend import prepare


MODEL_PATH = "data/saved_models/model.onnx"
OUT_PATH = "data/saved_models/model.engine"

# initialize TensorRT engine and parse ONNX model
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
config = builder.create_builder_config()

# we have only one image in batch
builder.max_batch_size = 1
# use FP16 mode if possible
# if builder.platform_has_fp16:
config.set_flag(trt.BuilderFlag.FP16)
# else:
    # print("fp16 False")

# parse ONNX
print('Beginning ONNX file parsing')
success = parser.parse_from_file(MODEL_PATH)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
print('Was parsing of ONNX file successful?: {}'.format(success))

# generate TensorRT engine optimized for the target platform
print('Building an engine...')
# engine = builder.build_cuda_engine(network)

# last_layer = network.get_layer(network.num_layers - 1)
# network.mark_output(last_layer.get_output(0))

serialized_engine = builder.build_serialized_network(network, config)
print("Completed creating Engine")

with open(OUT_PATH, "wb") as f:
    f.write(serialized_engine)

print("Saved engine")