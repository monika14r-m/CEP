# ---- Stage 1: train the model -----------------------------------------
# Downloads NSL-KDD and runs the full pipeline, so the image is genuinely
# self-building — no manual "run training, then build" step required.
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_pipeline.py .
RUN python train_pipeline.py

# ---- Stage 2: slim runtime image ---------------------------------------
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model_wrappers.py .
COPY --from=builder /app/artifacts ./artifacts

EXPOSE 8000
ENV ALLOWED_ORIGINS="*"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
