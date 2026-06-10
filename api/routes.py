from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SearchQuery(BaseModel):
    query: str

@router.post("/analyze-query")
def analyze_query(data: SearchQuery):

    query = data.query

    return {
        "query": query,
        "category": "Cyberbullying",
        "confidence": 94,
        "risk_level": "High",
        "reason": "Harmful insulting language detected."
    }