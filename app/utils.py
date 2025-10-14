import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import boto3
import os
import logging

logger = logging.getLogger(__name__)

BUCKET_NAME = "mlops-s3-20251005"
BASE_PATH = "distilbert-imdb"
LOCAL_DIR = "models_cache"


def download_from_s3(prefix: str):
    """Download model files from S3 to local cache directory."""
    s3 = boto3.client("s3")
    local_path = os.path.join(LOCAL_DIR, prefix)
    os.makedirs(local_path, exist_ok=True)

    logger.info(f"📥 Downloading model files from s3://{BUCKET_NAME}/{prefix}")
    result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    if "Contents" not in result:
        raise ValueError(f"No files found in {prefix}")

    for obj in result["Contents"]:
        key = obj["Key"]
        if key.endswith("/"):
            continue
        filename = os.path.basename(key)
        local_file = os.path.join(local_path, filename)
        if not os.path.exists(local_file):
            s3.download_file(BUCKET_NAME, key, local_file)
    return local_path


def load_model(version: str = "latest"):
    """Load model by version (v1, v2, or latest)."""
    if version == "latest":
        version = "v2"  # default to optimized version

    logger.info(f"🔍 Loading model version: {version}")
    prefix = f"{BASE_PATH}/{version}"
    model_dir = download_from_s3(prefix)

    # v1 = Hugging Face model
    if os.path.exists(os.path.join(model_dir, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info(f"✅ Loaded Hugging Face model from {version}")
        return model, tokenizer, version

    # v2 = quantized PyTorch model
    quantized_path = os.path.join(model_dir, "quantized_model.pth")
    if os.path.exists(quantized_path):
        model = torch.load(quantized_path, map_location="cpu")
        model.eval()

        # Try to load tokenizer from v1 if available
        v1_prefix = f"{BASE_PATH}/v1"
        try:
            tokenizer_dir = download_from_s3(v1_prefix)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            logger.info("🧠 Loaded tokenizer from v1 for quantized model.")
        except Exception:
            tokenizer = None
            logger.warning("⚠️ No tokenizer found. Quantized model may handle raw input.")

        logger.info(f"✅ Loaded quantized model from {version}")
        return model, tokenizer, version

    raise ValueError(f"No valid model found in {model_dir}")


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
