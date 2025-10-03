import os
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

model_dir = "models/distilbert_model"

# load model & tokenizer at start
try:
    tokenizer=AutoTokenizer.from_pretrained(model_dir)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
except Exception as e:
    # if model not found or failed to load, raise server error on startup
    raise RuntimeError(f"Failed to load model from {model_dir}: {e}")

@app.get("/ping")
def ping():
    return {"status":"ok"}

@app.get("/predict")
def predict(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    # simple CPU inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    score = outputs.logits.softmax(dim=-1).tolist()[0]
    label = int(outputs.logits.argmax().item())
    return {"label": label, "scores": score}
