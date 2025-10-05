import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
import os
import logging

# ------------- CONFIG -----------------
s3_bucket = 'mlops-s3-20251005'
s3_model_path = 'distilbert-imdb'
local_model_dir='models/distilbert-imdb'
# model_name=

# ------ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def download_model_from_s3():
    """Download model files from S3 if not already present locally."""
    if os.path.exists(local_model_dir):
        logger.info("Model already present locally.")
        return 
    os.makedirs(local_model_dir, exist_ok=True)
    s3 = boto3.client("s3")
    logger.info(f"Downloading model from s3://{s3_bucket}/{s3_model_path}....")
    paginator=s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=s3_bucket, Prefix=s3_model_path):
        for obj in result.get('Contents', []):
            key = obj['Key']
            local_path = os.path.join(local_model_dir, os.path.relpath(key, s3_model_path))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(s3_bucket, key, local_path)
            logger.info(f"Downloaded {key} to {local_path}")
    logger.info("Model downloaded")

# ---------- LOAD MODEL ----------
def load_model():
    """Load model and tokenizer from local or Hugging Face fallback."""
    try:
        download_model_from_s3()
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
        logger.info("Model loaded from local S3 copy.")
    except Exception as e:
        logger.warning(f"Falling back to Hugging Face model: {e}")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
        logger.info("Model loaded from Hugging Face.")
    return model, tokenizer

# ---- inference
def predict_text(model, tokenizer, text: str):
    """Perform sentiment prediction for input text."""
    inputs=tokenizer(text, return_tensors='pt',
                     truncation=True,padding=True)
    with torch.no_grad():
        outputs=model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
                
        return "positive" if predicted_class == 1 else "negative"

        