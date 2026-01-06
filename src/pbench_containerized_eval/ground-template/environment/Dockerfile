FROM python:3.13-slim

WORKDIR /app

RUN mkdir -p /logs/verifier

{install_pdf_tools}

COPY paper.pdf /app/paper.pdf
COPY task_meta.json /app/task_meta.json
