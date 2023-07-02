# Import FastAPI and other dependencies
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import json

# Create a FastAPI app
app = FastAPI()

# Load the image classifier model
model = tf.keras.models.load_model("newimageclassifier.h5")

# Define a custom class for handling numpy float32 objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# Define a route for uploading and classifying an image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Read the image file and convert it to a numpy array
    image_data = await file.read()
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Resize the image to 256x256 and normalize it
    resize = tf.image.resize(image, (256, 256))
    resize = resize / 255.0

    # Make a prediction using the model
    yhat = model.predict(np.expand_dims(resize, 0))

    # Return the prediction as a JSON response using the custom class
    return JSONResponse({"prediction": json.dumps(yhat[0][0], cls=NumpyEncoder)})