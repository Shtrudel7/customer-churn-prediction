FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends supervisor libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ui.py churn_model.joblib encodings.json ./
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8501
CMD ["/usr/bin/supervisord", "-n"]
