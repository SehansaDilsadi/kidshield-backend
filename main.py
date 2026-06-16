from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

class QueryRequest(BaseModel):
    query: str

@app.post("/analyze-query")
async def analyze_query(data: QueryRequest):

    query = data.query

    result = classifier(query)

    label = result[0]["label"]
    confidence = round(result[0]["score"] * 100, 2)

    # Risk Level Logic
    if confidence <= 40:
        risk_level = "Low"
    elif confidence <= 75:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Explanation Logic
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
        "toxic": "The query contains toxic or offensive language.",
        "insult": "The query may contain insulting language.",
        "threat": "The query may contain threatening language.",
        "obscene": "The query may contain inappropriate language.",
        "identity_hate": "The query may contain hateful content."
    }

    return explanations.get(
        label,
        "Potential harmful content detected."
    )