import asyncio
import aiohttp
import random
import time
from datetime import datetime

api_url = "http://52.13.56.115:8000/predict"
concurrent_users = 20
total_requests = 200

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


async def send_request(session, text):
    start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
        :-3
    ]  # e.g., 2025-10-07 15:45:12.123

    try:
        async with session.get(api_url, params={"text": text}) as response:
            result = await response.json()
            latency = (time.perf_counter() - start) * 1000  # ms
            print(
                f"[{timestamp}] Text: {text[:30]:<30} | Result: {result.get('sentiment', 'N/A'):<8} | Latency: {latency:.2f}ms"
            )
    except Exception as e:
        print(f"Error sending request: {e}")


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(total_requests):
            text = random.choice(sample_texts)
            task = asyncio.create_task(send_request(session, text))
            tasks.append(task)

            if len(tasks) >= concurrent_users:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
