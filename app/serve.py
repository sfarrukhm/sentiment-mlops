from fastapi import FastAPI, Request, Query as FastQuery
from pydantic import BaseModel
from app.utils import load_model, predict_text
import time
import logging
import os

# -----------------------------
# Logging Setup
# -----------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -----------------------------
# Initialize app and model
# -----------------------------
app = FastAPI(title="Sentiment Inference API")

# Load the base (non-quantized) model at startup
model, tokenizer, quantized = load_model(quantize=False)


class PredictRequest(BaseModel):
    text: str
    quantize: bool = False


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {"status": 200, "quantized": quantized}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    """
    Run sentiment prediction.
    If `quantize=True`, quantizes the model on the fly before inference.
    """
    start_time = time.time()

    global model, tokenizer, quantized

    # Handle quantization dynamically if requested
    if body.quantize != quantized:
        model, tokenizer, quantized = load_model(quantize=body.quantize)
        logger.info(f"⚙️ Switched quantization: quantize={quantized}")

    # Run prediction
    result = predict_text(model, tokenizer, body.text)
    latency = (time.time() - start_time) * 1000  # in ms

    client_ip = request.client.host
    logger.info(
        f"Client: {client_ip} | Quantized: {quantized} | Text: {body.text[:60]} | "
        f"Latency: {latency:.2f} ms | Result: {result}"
    )

    return {
        "sentiment": result,
        "latency_ms": f"{latency:.2f}",
        "quantized": quantized,
    }
