from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from utils.sqlite_db import increment_hit
app = FastAPI()
model = joblib.load("usecases/iris_usecase/model.joblib")

class InputSchema(BaseModel):

    Id: int

    SepalLengthCm: float

    SepalWidthCm: float

    PetalLengthCm: float

    PetalWidthCm: float


@app.post("/predict")
def predict(input: InputSchema):
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([input.dict()])
    increment_hit("iris_usecase")  


    # Predict using the loaded pipeline (which includes preprocessing + model)
    prediction = model.predict(input_df)[0]

    return {"prediction": prediction}
                        
@app.get("/health")
def health_check():
    return {"status": "ok", "usecase": "iris_usecase"}