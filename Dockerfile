# Use lightweight Python image
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Expose port for Cloud Run
ENV PORT=8080
EXPOSE 8080

# Run your Flask app
CMD ["python", "ingest_and_run.py"]
