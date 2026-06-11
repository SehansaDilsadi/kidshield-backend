from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="../trained_models/final_model",
    tokenizer="../trained_models/final_model"
)

def predict_risk(text):

    result = classifier(text)

    return result