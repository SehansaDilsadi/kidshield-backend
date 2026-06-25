from fastapi import FastAPI
from pydantic import BaseModel

from ai_model.predictor import predict_risk

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/analyze-query")
async def analyze_query(data: QueryRequest):

    query = data.query

    result = predict_risk(query)

    label = result["label"]

    confidence = round(result["score"] * 100, 2)

    # Risk level
    if label == "Safe":
        risk_level = "Low"
    elif label == "Toxic":
        risk_level = "Medium"
    elif label == "Cyberbullying":
        risk_level = "High"
    elif label == "Hate Speech":
        risk_level = "High"
    else:
        risk_level = "Medium"

    explanation = generate_explanation(label)

    return {
        "query": query,
        "category": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation
    }

def generate_explanation(label):

    explanations = {
        "Safe": "The search appears safe.",
        "Toxic": "The search contains offensive language.",
        "Cyberbullying": "The search may contain bullying or harassment.",
        "Hate Speech": "The search may contain hateful content."
    }

    return explanations.get(
        label,
        "Potential harmful content detected."
    )