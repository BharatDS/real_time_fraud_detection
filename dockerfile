FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "--version"]