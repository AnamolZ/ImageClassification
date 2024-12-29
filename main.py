import io
import torch
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

yolov5_model = load_yolov5_model()
classifier_model = tf.keras.models.load_model('classifier_model.h5')

class FurnishingResponse(BaseModel):
    Type: str
    Furniture: Union[str, None]

def classify_image_type(image, classifier_model):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = classifier_model.predict(img_array)
    is_interior = prediction[0] <= 0.5
    return "Interior" if is_interior else "Exterior"

def detect_furniture(image, model):
    results = model(image)
    detections = results.pandas().xyxy[0]
    furniture_classes = ["chair", "sofa", "bed", "table", "desk", "cabinet"]
    furniture_detected = detections[detections['name'].isin(furniture_classes)]
    return furniture_detected

def classify_furnishing(image, model):
    furniture_detected = detect_furniture(image, model)
    is_furnished = len(furniture_detected) > 0
    furnishing_status = {
        "Furniture": "Furnished" if is_furnished else "Unfurnished",
    }
    return furnishing_status

@app.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={"message": "Everything is good here 👀"})

@app.post("/classify", response_model=FurnishingResponse)
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))

    image_type = classify_image_type(image, classifier_model)
    
    if image_type == "Interior":
        furnishing_status = classify_furnishing(image, yolov5_model)
        result = {"Type": image_type, "Furniture": furnishing_status["Furniture"]}
    else:
        result = {"Type": image_type, "Furniture": None}
    
    return result