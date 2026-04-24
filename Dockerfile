FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner_gemini.py .

ENV TG_TOKEN=""
ENV TG_CHAT_ID=""
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "scanner_gemini:app", "--worker-class", "gthread", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "--preload"]
