import os
import torch
import logging
import boto3
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.ao.quantization import quantize_dynamic

logger = logging.getLogger(__name__)

S3_BUCKET = "mlops-s3-20251005"
S3_BASE_PATH = "distilbert-imdb/v1"  # always load fine-tuned model from v1 path
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")


def download_from_s3(s3_client, s3_path, local_path):
    """Download a single file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(S3_BUCKET, s3_path, local_path)


def load_model(quantize: bool = False):
    """
    Loads the fine-tuned DistilBERT model from S3.
    If `quantize=True`, applies dynamic quantization for faster CPU inference.
    """
    logger.info(f"🔍 Loading fine-tuned DistilBERT model (quantize={quantize})")
    s3 = boto3.client("s3")

    local_model_path = os.path.join(LOCAL_MODEL_DIR, "v1")
    os.makedirs(local_model_path, exist_ok=True)

    # Model files to fetch if missing
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
            logger.info(f"📥 Downloading {f} from S3...")
            download_from_s3(s3, f"{S3_BASE_PATH}/{f}", local_file)

    # Load tokenizer + base fine-tuned model
    tokenizer = DistilBertTokenizer.from_pretrained(local_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(local_model_path)

    # Optionally apply quantization
    if quantize:
        logger.info("⚙️ Applying dynamic quantization for inference...")
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info("✅ Quantized model ready.")

    model.eval()
    logger.info("✅ Model loaded and set to eval mode.")
    return model, tokenizer, quantize


def predict_text(model, tokenizer, text: str):
    """Run inference on text using the provided model and tokenizer."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()
        return "positive" if pred == 1 else "negative"

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "error"
