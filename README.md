# 🚀 Sentiment Analysis MLOps Pipeline

An **end-to-end MLOps project** demonstrating model deployment, observability, and performance optimization using **FastAPI**, **AWS S3**, and **GitHub Actions**.

This repository shows how to take an NLP model from **fine-tuning to scalable inference** — with dynamic quantization and CI/CD automation for real-world readiness.

---

## 🧠 Project Overview

This project serves a fine-tuned **DistilBERT sentiment classifier** through a production-grade API.
It includes model loading from **AWS S3**, optional **quantized inference**, structured **logging**, **load testing**, and **CI/CD automation**.

Quantization reduced average latency by **≈60%**, proving the practical value of lightweight model optimization.

| Mode                 | Avg Latency (ms) | P95 Latency (ms) | Improvement   |
| -------------------- | ---------------- | ---------------- | ------------- |
| Without Quantization | 3274.15          | 5912.65          | —             |
| With Quantization    | **1302.57**      | **2581.50**      | ✅ ~60% faster |

---

## 🧩 Tech Stack

| Category         | Tools Used                                 |
| ---------------- | ------------------------------------------ |
| **Modeling**     | Hugging Face Transformers (DistilBERT)     |
| **Serving**      | FastAPI, Uvicorn                           |
| **Deployment**   | AWS EC2, S3                                |
| **Automation**   | GitHub Actions (CI/CD)                     |
| **Monitoring**   | Custom structured logging (`logs/app.log`) |
| **Testing**      | Pytest                                     |
| **Load Testing** | Async load simulator with aiohttp          |
| **Optimization** | PyTorch Dynamic Quantization               |

---

## ⚙️ Key Features

* 🔁 **Automated model fetch from S3** during startup
* ⚙️ **Dynamic quantization toggle** for faster CPU inference
* 📈 **Structured request logging** (latency, client IP, text length, sentiment)
* 🧪 **Pytest-based CI pipeline** for stability
* 🌐 **FastAPI endpoint** for real-time predictions
* 📊 **Load simulator** to measure performance under concurrent requests

---

## 🧰 API Usage

### **Health Check**

```bash
GET /ping
```

✅ Response:

```json
{"status": 200, "quantized": false}
```

### **Predict Endpoint**

```bash
POST /predict
```

**Body:**

```json
{
  "text": "The movie was absolutely fantastic!",
  "quantize": true
}
```

✅ Response:

```json
{
  "sentiment": "positive",
  "latency_ms": "1320.45",
  "quantized": true
}
```

---

## 📦 Project Structure

```
sentiment-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration: pytest, lint checks
│       └── cd.yml              # Continuous Deployment: deploy to EC2
│
├── app/
│   ├── serve.py                # FastAPI app for serving predictions
│   └── utils.py                # Model loading, quantization, and inference logic
│
├── logs/
│   ├── app.log                 # Application logs
│   ├── latency-stats.txt       # Performance summary
│   └── simulator.log           # Load test results
│
├── scripts/
│   ├── load_simulator.py       # Simulates concurrent requests for stress testing
│   └── analyze_simulator_logs.py # Parses and visualizes latency results
│
├── deploy.sh                   # Shell script for EC2 deployment
├── Makefile                    # Unified dev commands (test, run, deploy)
├── requirements.txt             # Project dependencies
├── test_setup.py                # Basic API and health-check tests
└── README.md                    # Documentation (you’re reading it!)

```

---

## 🔄 CI/CD Pipeline

GitHub Actions automates:

* ✅ Environment setup (Python + dependencies)
* ✅ Linting & testing via `pytest`
* ✅ Failure alerts on PRs
* ✅ (Optional) Deployment to EC2 after passing tests

This ensures your API is **always tested before merging**, just like real MLOps production pipelines.

---

## 📊 Load Testing

Run async load simulation to test stability under concurrent requests:

```bash
python scripts/load_simulator.py
```

Results are logged to `logs/simulator.log` and visualized with a latency-over-time plot.

---

## ⚡ Quantization Impact

| Metric               | Without Quantization | With Quantization |
| -------------------- | -------------------- | ----------------- |
| **Min Latency (ms)** | 331.52               | 291.97            |
| **Avg Latency (ms)** | 3274.15              | **1302.57**       |
| **P95 Latency (ms)** | 5912.65              | **2581.50**       |
| **Max Latency (ms)** | 9143.29              | **4830.67**       |

👉 Demonstrates how **PyTorch dynamic quantization** reduces model size and speeds up inference — essential for CPU-based deployments.

---

## 🧪 Run Tests

```bash
make test
```

or

```bash
PYTHONPATH=. pytest -v
```

---

## ☁️ Deployment

The app is designed for **AWS EC2 deployment**.
Model files are fetched from **S3** on first run and cached locally.

```bash
uvicorn app.serve:app --host 0.0.0.0 --port 8000
```

---

## 🧠 MLOps Pitch

This project demonstrates a **production-ready NLP deployment pipeline** with performance optimization and automation at its core.
By integrating **AWS**, **FastAPI**, and **GitHub Actions**, it showcases how MLOps turns research models into **reliable, scalable services** — cutting latency by **60%** through smart model quantization.

---

## 👨‍💻 Author

**M. Farrukh Mehmood**  
📫 [smfarrukhm@gmail.com](mailto:smfarrukhm@gmail.com)  

🔗 [LinkedIn](https://www.linkedin.com/in/sfarrukhm) | [GitHub](https://github.com/sfarrukhm)
