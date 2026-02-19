# 1. Gunakan image dasar Python yang ringan
FROM python:3.9-slim

# 2. Tentukan directory kerja di dalam container
WORKDIR /app

# 3. Copy file requirements dan instal library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy folder yang dibutuhkan (app dan models)
COPY ./app ./app
COPY ./models ./models

# 5. Expose port yang digunakan FastAPI
EXPOSE 8000

# 6. Jalankan server saat container dimulai
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]