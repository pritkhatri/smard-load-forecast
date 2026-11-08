# API image. Build after `make data train`, so that models/ and data/processed/ exist:
#   docker build -t smardfc .
#   docker run -p 8000:8000 smardfc
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY models/model.joblib models/model.joblib
COPY data/processed/load.parquet data/processed/load.parquet
ENV SMARDFC_MODEL=models/model.joblib SMARDFC_DATA=data/processed/load.parquet

EXPOSE 8000
CMD ["uvicorn", "smardfc.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
