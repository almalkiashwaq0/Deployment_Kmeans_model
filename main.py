from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel


app = FastAPI()
 # GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}




app = FastAPI()
model = joblib.load("knn_model.joblib")

class InputFeatures(BaseModel):
    appearance: int
    goals: float
    assists: float
    goals_conceded: float
    minutes_played: int
    days_injured: int
    games_injured: int
    award: int
    highest_value: int
    position_Goalkeeper: bool

def preprocessing(input_features: InputFeatures):
    return {
        'appearance': input_features.appearance,
        'goals': input_features.goals,
        'assists': input_features.assists,
        'goals_conceded': input_features.goals_conceded,
        'minutes_played': input_features.minutes_played,
        'days_injured': input_features.days_injured,
        'games_injured': input_features.games_injured,
        'award': input_features.award,
        'highest_value': input_features.highest_value,
        'position_Goalkeeper': input_features.position_Goalkeeper
    }

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict([list(data.values())])  # Ensure data is in the correct format
    return {"prediction": y_pred.tolist()[0]}
