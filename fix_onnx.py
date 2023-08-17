from os.path import join

import onnx_graphsurgeon as gs
import onnx
from onnx_tf.backend import prepare
import numpy as np


MODEL_PATH = "data/saved_models/model.onnx"
OUT_PATH = "data/saved_models/model_fp32.onnx"

onnx_model = onnx.load(MODEL_PATH)

graph = gs.import_onnx(onnx_model)
for inp in graph.inputs:
    inp.dtype = np.float32

onnx.save(gs.export_onnx(graph), OUT_PATH)