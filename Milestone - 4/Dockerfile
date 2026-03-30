FROM python:3.10

WORKDIR /app

# ✅ Install system libraries (IMPORTANT FIX)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]