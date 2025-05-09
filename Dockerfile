# Build stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsndfile1 \
#     && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm
RUN python -m textblob.download_corpora

# Copy the project files
COPY src/ src/

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]