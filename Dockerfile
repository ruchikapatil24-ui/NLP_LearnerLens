# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend with FastAPI, NLTK & spaCy
FROM python:3.11-slim
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora and spaCy language model
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"
RUN python -m spacy download en_core_web_sm

# Copy backend application source
COPY backend/app ./app

# Copy compiled frontend distribution to static directory
COPY --from=frontend-builder /app/frontend/dist ./static

# Environment settings
ENV PORT=8000
EXPOSE 8000

# Start unified server serving API and Frontend from single port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
