from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

MODEL = keras.Sequential([
    keras.layers.TFSMLayer("../saved_models/1", call_endpoint="serving_default")
])

#endpoint = 'http://localhost:8501/v1/models/potato_classification_model:predict'

CLASS_NAMES = ["Early Blight" , "Healthy", "Late Blight" ]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction['dense_1'][0])]
    confidence = np.max(np.array(prediction['dense_1'][0]))
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

