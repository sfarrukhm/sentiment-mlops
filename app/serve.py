from fastapi import FastAPI, Request, Query
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
app = FastAPI(title="Sentiment Inference API with Versioning")

model, tokenizer, current_version = load_model()

@app.get("/ping")
def ping():
    return {"status": 200, "model_version": current_version}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.get("/predict")
async def predict(request: Request, text: str = Query(...)):
    """Run sentiment prediction."""
    start_time = time.time()
    result = predict_text(model, tokenizer, text)
    latency = (time.time() - start_time) * 1000  # ms

    client_ip = request.client.host
    logger.info(
        f"Client: {client_ip} | Version: {current_version} | Text: {text[:60]} | Latency: {latency:.2f} ms | Result: {result}"
    )

    return {"sentiment": result, "latency_ms": f"{latency:.2f}", "model_version": current_version}

# -----------------------------
# Model Switch Endpoint
# -----------------------------
@app.post("/switch_model")
def switch_model(version: str):
    """Hot-swap model version without restarting the API."""
    global model, tokenizer, current_version

    try:
        new_model, new_tokenizer, loaded_version = load_model(version)
        model, tokenizer, current_version = new_model, new_tokenizer, loaded_version
        logger.info(f"✅ Switched to model version: {current_version}")
        return {"message": f"Model switched to {current_version}"}
    except Exception as e:
        logger.error(f"❌ Model switch failed: {e}")
        return {"error": str(e)}

