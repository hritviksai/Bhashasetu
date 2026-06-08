# Use slim Python image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by torch/transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces requires the app to listen on port 7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
