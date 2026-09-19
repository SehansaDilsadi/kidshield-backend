from transformers import pipeline

_MODEL_PATH = "SehansaDilsadi/kidshield-distilbert"

classifier = pipeline(
    "text-classification",
    model=_MODEL_PATH,
    tokenizer=_MODEL_PATH
)

# Exposed so attribution.py can run Integrated Gradients against the same
# in-memory model/tokenizer instead of loading a second copy.
model = classifier.model
tokenizer = classifier.tokenizer

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
        "score": score,
        "label_id": model.config.label2id[label],
    }