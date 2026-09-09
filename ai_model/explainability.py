"""Explainability (XAI) layer for KidShield search classification.

Uses keyword patterns derived from the ai_training dataset to identify
influential words and generate parent-friendly explanations.
"""

import re
from typing import List

# Keyword signals aligned with ai_training dataset label patterns
CATEGORY_KEYWORDS = {
    "Cyberbullying": [
        "bully", "bullying", "harass", "harassment", "loser", "stupid", "ugly",
        "fat", "classmate", "victim", "mock", "tease", "rumor", "excluded",
        "worthless", "nobody likes", "kill yourself", "kys", "lame", "pathetic",
    ],
    "Hate Speech": [
        "hate", "racist", "racism", "slur", "nazi", "supremacist", "discriminat",
        "bigot", "sexist", "homophob", "transphob", "antisemit", "retard",
    ],
    "Violent Content": [
        "kill", "murder", "weapon", "gun", "bomb", "stab", "shoot", "attack",
        "violent", "assault", "fight", "hurt", "blood", "death", "strangle",
    ],
    "Inappropriate Content": [
        "porn", "xxx", "nude", "naked", "sex", "drug", "alcohol", "weed",
        "cocaine", "high", "suicide", "self harm", "cutting", "meth",
    ],
    "Toxic": [
        "damn", "hell", "crap", "idiot", "moron", "fuck", "shit", "bitch",
        "asshole", "offensive", "vulgar", "suck", "whore",
    ],
}

RISK_BY_CATEGORY = {
    "Safe": "Low",
    "Toxic": "Medium",
    "Inappropriate Content": "Medium",
    "Cyberbullying": "High",
    "Hate Speech": "High",
    "Violent Content": "High",
}

SUGGESTED_ACTIONS = {
    "Safe": "No action needed — query logged for your records.",
    "Toxic": "Review the search with your child and discuss appropriate online language.",
    "Inappropriate Content": "Have a conversation about age-appropriate content and set browsing guidelines.",
    "Cyberbullying": "Discuss respectful online behaviour with your child.",
    "Hate Speech": "Address discriminatory language immediately and explain why it is harmful.",
    "Violent Content": "Talk to your child about violence and ensure they feel safe. Consider professional support if needed.",
}

EXPLANATIONS = {
    "Cyberbullying": "The query contains language suggesting intentional harassment of another individual.",
    "Hate Speech": "The query contains hateful or discriminatory language targeting individuals or groups.",
    "Violent Content": "The query references violence, weapons, or harmful acts toward others.",
    "Inappropriate Content": "The query seeks mature or inappropriate content not suitable for children.",
    "Toxic": "The query contains offensive or toxic language according to the AI training dataset.",
    "Safe": "The search appears safe and appropriate according to the AI training dataset.",
}


def refine_category(model_label: str, query: str) -> str:
    """Refine the model label using dataset keyword signals."""
    if model_label == "Safe":
        return "Safe"

    lower = query.lower()

    if model_label == "Cyberbullying":
        return "Cyberbullying"
    if model_label == "Hate Speech":
        return "Hate Speech"

    if model_label == "Toxic":
        violent_hits = sum(1 for kw in CATEGORY_KEYWORDS["Violent Content"] if kw in lower)
        inappropriate_hits = sum(1 for kw in CATEGORY_KEYWORDS["Inappropriate Content"] if kw in lower)
        if violent_hits >= 1:
            return "Violent Content"
        if inappropriate_hits >= 1:
            return "Inappropriate Content"
        return "Toxic"

    return model_label


def extract_key_words(query: str, category: str) -> List[str]:
    """Identify influential words that drove the classification."""
    if category == "Safe":
        return []

    lower = query.lower()
    primary = CATEGORY_KEYWORDS.get(category, [])
    found: List[str] = []

    for kw in sorted(primary, key=len, reverse=True):
        if kw in lower and kw not in found:
            found.append(kw)

    tokens = re.findall(r"\b\w+\b", lower)
    for token in tokens:
        if len(token) <= 3 or token in found:
            continue
        for kw in primary:
            if token == kw or (len(kw) > 4 and kw in token):
                found.append(token)
                break

    return found[:5]


def get_risk_level(category: str) -> str:
    return RISK_BY_CATEGORY.get(category, "Medium")


def get_suggested_action(category: str) -> str:
    return SUGGESTED_ACTIONS.get(category, "Review this search with your child.")


def generate_explanation(category: str, key_words: List[str]) -> str:
    return EXPLANATIONS.get(
        category,
        "Potential harmful content detected according to the AI training dataset.",
    )


def analyze_with_xai(model_label: str, query: str) -> dict:
    category = refine_category(model_label, query)
    key_words = extract_key_words(query, category)
    risk_level = get_risk_level(category)
    explanation = generate_explanation(category, key_words)
    suggested_action = get_suggested_action(category)

    return {
        "category": category,
        "risk_level": risk_level,
        "key_words": key_words,
        "explanation": explanation,
        "suggested_action": suggested_action,
    }
