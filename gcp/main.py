import os
import tensorflow as tf
from google.cloud import storage
from PIL import Image
import numpy as np

BUCKET_NAME = "srinivasaperisetla-models"
DOWNLOAD_DIR = "models/deployment_model_final.keras"
MODEL_DIR = "/tmp/deployment_model_final.keras"
class_names = ["Early Blight", "Healthy", "Late Blight"]

local_model_dir = "saved_models/deployment_model_final.keras"

model = None

def load_model(dir):
    model_sample = tf.keras.models.load_model(dir)
    return model_sample


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def download_model_from_gcs(bucket_name, model_folder):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_folder)

    for blob in blobs:
        file_path = os.path.join(MODEL_DIR, blob.name[len(model_folder):])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
        print(f"Downloaded {blob.name} to {file_path}. Size: {os.path.getsize(file_path)} bytes")


def predict(request):
    global model
    if model is None:
        print("Model is not loaded, downloading model...")
        
        # Download the entire model directory
        download_blob(BUCKET_NAME, DOWNLOAD_DIR, MODEL_DIR)

        #download_model_from_gcs(BUCKET_NAME, MODEL_DIR)
        
        # Verify the downloaded files
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                print(f"File: {os.path.join(root, file)}, Size: {os.path.getsize(os.path.join(root, file))} bytes")

        # Load the model from the directory
        try:
            '''
            model = tf.keras.Sequential([
                tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
            ])
            '''
            model = tf.keras.models.load_model(MODEL_DIR)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return {"error": str(e)}

    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image / 255.0  # Normalize the image
    img_array = tf.expand_dims(image, 0)

    predictions = model.predict(img_array)
    print(predictions)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return {"class": predicted_class, "confidence": confidence}


'''
hello = load_model(local_model_dir)
print(hello.summary())
'''
