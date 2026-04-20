FROM python:3.13-slim

WORKDIR /app

RUN mkdir -p /logs/verifier

{install_pdf_tools}

# Verifier Python deps required by /tests/check_prediction.py (embedding +
# LLM-based property matching, pymatgen-based formula scoring).
RUN pip install --no-cache-dir \
    google-genai \
    numpy \
    pandas \
    pydantic \
    pymatgen \
    pyyaml \
    'scikit-learn>=1.8.0' \
    tenacity \
    tqdm

# Bundle the monorepo-local llm_utils package so check_prediction.py can import
# it. prepare_harbor_tasks.py copies src/llm_utils into this build context.
# GOOGLE_API_KEY is forwarded into the verifier at runtime via [verifier.env]
# in task.toml; no ENV directive is needed here.
COPY llm_utils /opt/llm_utils
ENV PYTHONPATH=/opt

COPY paper.pdf /app/paper.pdf
COPY task_meta.json /app/task_meta.json
