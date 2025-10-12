#!/bin/bash
cd /home/ubuntu/sentiment-mlops

# Pull new code
git pull origin main

# Activate venv
source .venv-smlops/bin/activate

pip install -r requirements.txt

# Restart FastAPI app
sudo systemctl restart sentiment.service
