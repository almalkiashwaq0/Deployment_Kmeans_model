from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"


model = joblib.load("kmeans_model.joblib")
scaler = joblib.load("scaler_modell.joblib")

class InputFeatures(BaseModel):
    appearance: int
    goals: float
    award: int
    height: float

def preprocessing(input_features: InputFeatures):
    dict_f =  {
        'appearance': input_features.appearance,
        'goals': input_features.goals,
        'award': input_features.award,
        'height': input_features.height,
    }
     
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    scaled_features = scaler.transform([list(dict_f.values
 ())])

    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}

