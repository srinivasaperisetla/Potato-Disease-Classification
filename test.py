import numpy as np
import tensorflow as tf

local_model_dir = "saved_models/2.h5"

model  = None
'''
try:
    model = tf.keras.models.load_model(local_model_dir)
    print("Model loaded successfully.")

    model =  tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(local_model_dir, call_endpoint="serving_default")
    ])
'''

print(tf.__version__)

try:
    model = tf.keras.models.load_model(local_model_dir)
    print(model.summary())
except Exception as e:
    print(f"Failed to load model: {e}", model)


