FROM python:3.13-slim
# Minimal environment for search task (no PDF tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
