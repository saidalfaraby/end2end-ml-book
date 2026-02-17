# Inisialisasi Project & Data Versioning (DVC)

Tujuan utama kita adalah membangun **jalur pipa (pipeline)** yang otomatis. Dengan struktur ini, ketika ada data baru (Cycle 2), proses dari *training* hingga *deployment* dapat berjalan dengan intervensi manual yang minim.

## 1. Persiapan Environment (Conda)

Gunakan **Conda** untuk membuat lingkungan terisolasi agar tidak mengganggu pustaka lain di sistem Anda.

```bash
# Membuat environment baru dengan python 3.9
conda create -n mlops-tutorial python=3.9 -y
conda activate mlops-tutorial

# Install library utama melalui conda dan pip
conda install pandas scikit-learn -y
pip install dvc dvc-gdrive mlflow fastapi uvicorn
```