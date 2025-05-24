FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Затем копируем остальной код
COPY . .


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]