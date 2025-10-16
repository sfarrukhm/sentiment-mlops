import asyncio
import aiohttp
import random
import time
import logging
import os
from datetime import datetime
from tqdm import tqdm  # progress bar

# -----------------------------
# Config
# -----------------------------
api_url = "http://52.13.56.115:8000/predict"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "simulator.log")

# delete previous log file if it exists
if os.path.exists(log_path):
    os.remove(log_path)
# -----------------------------
# Setup logging
# -----------------------------
logger = logging.getLogger("simulator")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


sample_texts = [
    "An absolute masterpiece. Every frame felt alive and meaningful.",
    "Pretty boring, I fell asleep halfway through.",
    "The acting was strong, but the story was slow and predictable.",
    "A surprisingly emotional journey — I didn’t expect to cry!",
    "Terrible script. Not even the great cast could save it.",
    "Visually stunning, with music that fits perfectly into every scene.",
    "One of the worst films I’ve seen in years. No plot, no effort.",
    "Short, sweet, and hilarious. Perfect for a weekend watch.",
    "The movie tried too hard to be deep, but ended up being confusing.",
    "Loved it! The chemistry between the leads was amazing.",
    "It’s okay — not great, not terrible. Just average entertainment.",
    "Beautiful cinematography but zero emotional connection.",
    "A complete mess from start to finish. I wanted to leave the theater.",
    "Unexpectedly good! I went in with low expectations and was blown away.",
    "Poorly edited and badly paced. Felt three hours long.",
    "Heartwarming and funny. Reminded me of the classic comedies from the 90s.",
    "Good ideas, but the execution was flat and lifeless.",
    "The twists were brilliant! I didn’t see any of them coming.",
    "Cringe-worthy dialogue and wooden performances.",
    "A solid film with great character development and a satisfying ending.",
    "Honestly, I don’t get the hype. It was just fine.",
    "The director’s best work yet — thrilling, clever, and beautifully shot.",
    "Predictable ending but still enjoyable to watch with family.",
    "The humor was forced, and most of the jokes didn’t land.",
    "Incredible performances by the entire cast. Totally gripping.",
    "A dull and uninspired remake that adds nothing new.",
    "A moving story told with honesty and heart. Highly recommended.",
    "Loud, messy, and overlong. I couldn’t wait for it to end.",
    "Great soundtrack, mediocre story, but entertaining overall.",
    "A flawed but fascinating film that lingers in your mind afterward.",
]


# -----------------------------
# Async request
# -----------------------------
async def send_request(session, text,quantize):
    start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    payload={"text": text,'quantize':quantize}

    try:
        async with session.post(api_url, json=payload) as response:
            result = await response.json()
            latency = (time.perf_counter() - start) * 1000  # ms

            # log only to file
            logger.info(
                f"[{timestamp}] Text: {text[:30]:<30} | Result: {result.get('sentiment', 'N/A'):<8} | Q:{result.get('quantized')} | Latency: {latency:.2f}ms"
            )
    except Exception as e:
        logger.error(f"Error: {e}")


# -----------------------------
# Main function with progress bar
# -----------------------------
async def main(concurent_users, total_requests,quantize):
    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm(total=total_requests, desc="Simulating load", ncols=80) as pbar:
            for i in range(total_requests):
                text = random.choice(sample_texts)
                task = asyncio.create_task(send_request(session, text,quantize))
                tasks.append(task)

                if len(tasks) >= concurrent_users:
                    await asyncio.gather(*tasks)
                    tasks = []
                    pbar.update(concurrent_users)

            if tasks:
                await asyncio.gather(*tasks)
                pbar.update(len(tasks))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load Simulator for Sentiment API")
    parser.add_argument('--concurrent', type=int, default=1, help='Number of concurrent users')
    parser.add_argument('--requests', type=int, default=1, help='Total number of requests to send')
    parser.add_argument('--quantize',type=str,default="true", help="Whether to use quantized model or fp32 model")
    args=parser.parse_args()
    concurrent_users = args.concurrent
    total_requests = args.requests
    quantize = args.quantize == "true"

    asyncio.run(main(concurrent_users, total_requests,quantize))
    