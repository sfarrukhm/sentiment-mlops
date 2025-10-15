import os
import torch
import logging
import boto3
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

logger = logging.getLogger(__name__)

S3_BUCKET = "mlops-s3-20251005"
S3_BASE_PATH = "distilbert-imdb"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")


def download_from_s3(s3_client, s3_path, local_path):
    """Download a single file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(S3_BUCKET, s3_path, local_path)


def load_model(version: str = "v2"):
    """
    Loads a model version from S3.
    - v1: full Hugging Face model (config + tokenizer + weights)
    - v2: quantized PyTorch model (entire model saved)
    """
    logger.info(f"🔍 Loading model version: {version}")
    s3 = boto3.client("s3")

    local_model_path = os.path.join(LOCAL_MODEL_DIR, version)
    os.makedirs(local_model_path, exist_ok=True)

    if version == "v1":
        files = [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]
        for f in files:
            local_file = os.path.join(local_model_path, f)
            if not os.path.exists(local_file):
                download_from_s3(s3, f"{S3_BASE_PATH}/{version}/{f}", local_file)

        tokenizer = DistilBertTokenizer.from_pretrained(local_model_path)
        model = DistilBertForSequenceClassification.from_pretrained(local_model_path)

    elif version == "v2":
        local_quant_path = os.path.join(local_model_path, "quantized_model.pth")
        if not os.path.exists(local_quant_path):
            logger.info("📥 Downloading full quantized model from S3...")
            download_from_s3(s3, f"{S3_BASE_PATH}/{version}/quantized_model.pth", local_quant_path)

        logger.info("⚙️ Loading full quantized DistilBERT model...")
        model = torch.load(local_quant_path, map_location="cpu")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    else:
        raise ValueError(f"Unknown model version: {version}")

    model.eval()
    logger.info(f"✅ Model {version} loaded and set to eval mode.")
    return model, tokenizer, version




def predict_text(model, tokenizer, text: str):
    """Run inference with automatic handling for both model types."""
    try:
        if tokenizer:  # Hugging Face model or quantized with tokenizer
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
        else:  # Custom quantized model expecting raw input tensor
            if isinstance(text, str):
                # naive fallback if no tokenizer exists
                input_tensor = torch.tensor([[ord(c) % 256 for c in text[:256]]])
                with torch.no_grad():
                    logits = model(input_tensor.float())
            else:
                logits = model(text)

        pred = torch.argmax(logits, dim=1).item()
        return "positive" if pred == 1 else "negative"

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "error"
