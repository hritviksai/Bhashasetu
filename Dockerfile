# Use slim Python image for smaller build size
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

# Render.com dynamically assigns a PORT environment variable
# Default to 10000 if not set (Render's typical default)
EXPOSE 10000

# Run the Flask app — reads PORT from environment
CMD ["python", "app.py"]
