import tensorflow as tf

MODEL_PATH = (
    "data/saved_models/tf"
)
OUT_PATH = "data/saved_models/tf/model.tflite"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)
tflite_model = converter.convert()

# Save the model
with open(OUT_PATH, 'wb') as f:
    f.write(tflite_model)