FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY churn_model.joblib .
COPY encodings.json .
COPY app.py .
COPY ui.py .

# Устанавливаем зависимости и системные библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# ВОТ ЗДЕСЬ МЫ ДОБАВИЛИ libgomp1 (и чистим кэш для уменьшения размера)
RUN apt-get update && apt-get install -y supervisor libgomp1 && rm -rf /var/lib/apt/lists/*

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8501

CMD ["/usr/bin/supervisord"]