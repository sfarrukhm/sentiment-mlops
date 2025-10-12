from fastapi import FastAPI, Request
from app.utils import load_model, predict_text


import time
import logging
import os


# create file handler
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)

# get root logger and attach
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Initialize app and model ---
app = FastAPI(title="Sentiment Inference API")
model, tokenizer = load_model()


# ---- Check connectivity
@app.get("/ping")
def ping():
    return {"status": 200}


@app.get("/predict")
async def predict(request: Request, text: str):
    """API endpoint for sentiment prediction."""
    start_time = time.time()
    result = predict_text(model, tokenizer, text)
    latency = (time.time() - start_time) * 1000  # ms

    client_ip = request.client.host
    logger.info(
        f"Client: {client_ip} | Text Length: {len(text)} | Latency: {latency: .2f} ms | Result: {result}"
    )
    return {"sentiment": result, "latency_ms": latency}
