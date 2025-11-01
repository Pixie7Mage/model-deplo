# app.py

import os
# ✅ Disable GPU so TensorFlow/Keras doesn’t try to use CUDA on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input format
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Model is live on Render!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to numpy array
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
