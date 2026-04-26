FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scanner.py .
ENV TG_TOKEN=""
ENV TG_CHAT_ID=""
ENV OPENROUTER_API_KEY=""
ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn", "scanner:app", "--worker-class", "gthread", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "--preload"]
# updated
