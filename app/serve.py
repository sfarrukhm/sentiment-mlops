import os
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.utils import load_model, predict_text


import time
import logging
logging.basicConfig(filename='app.log',
                    level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s'
                    )

# --- Initialize app and model ---
app = FastAPI(title="Sentiment Inference API")
model, tokenizer = load_model()
@app.get("/predict")
async def predict(request: Request, text:str)
    """API endpoint for sentiment prediction."""
    strat_time=time.time()
    result=predict_text(model, tokenizer, text)
    latency=(time.time()-start_time)*1000 # ms

    client_ip = request.client.host
    logging.info(
    f"Client: {client_ip} | Text Length: {len(text)} | Latency: {latency: .2f} ms | Result: {result}")
    return {"sentiment":result, "latency_ms":latency}
