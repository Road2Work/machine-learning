FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables untuk mencegah Python menulis file .pyc dan memastikan output langsung ke console
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependensi sistem yang dibutuhkan oleh machine learning dan audio processing (ffmpeg, dsb)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt terlebih dahulu (untuk memanfaatkan cache Docker)
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

# Expose port yang digunakan oleh FastAPI
EXPOSE 8000

# Jalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
