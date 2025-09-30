# Gunakan base image Python versi slim
FROM python:3.12-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt terlebih dahulu untuk caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project ke container
COPY . .

# Expose port untuk Railway / Docker run
EXPOSE 8000

# Jalankan Flask pakai Gunicorn (production-ready)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
