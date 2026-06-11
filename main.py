from fastapi import FastAPI
from pydantic import BaseModel
from ai_model.predictor import predict_risk

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

@app.post("/analyze")
def analyze(data: SearchQuery):

    result = predict_risk(data.query)

    return {
        "query": data.query,
        "prediction": result
    }