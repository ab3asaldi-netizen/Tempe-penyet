FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner.py .

ENV TG_TOKEN=""
ENV TG_CHAT_ID=""
ENV PORT=8080

EXPOSE 8080

CMD ["python", "scanner:py"]
