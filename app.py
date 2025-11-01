# app.py
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi import FastAPI


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
