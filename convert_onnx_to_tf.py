from os.path import join

import onnx
from onnx_tf.backend import prepare


MODEL_PATH = (
    "data/saved_models/model.onnx"
)
OUT_PATH = "data/out"

onnx_model = onnx.load(MODEL_PATH)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(join(OUT_PATH,'tf','model'))