from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="../trained_models/final_model",
    tokenizer="../trained_models/final_model"
)

LABEL_MAP = {
    "LABEL_0": "Safe",
    "LABEL_1": "Toxic",
    "LABEL_2": "Cyberbullying",
    "LABEL_3": "Hate Speech"
}

def predict_risk(text):

    result = classifier(text)

    label = result[0]["label"]
    score = result[0]["score"]

    return {
        "label": LABEL_MAP.get(label, label),
        "score": score
    }