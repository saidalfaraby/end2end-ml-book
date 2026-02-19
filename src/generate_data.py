import kagglehub
import pandas as pd
import os
import shutil

def ingest_data():
    print("Mengunduh dataset dari Kaggle...")
    # Mengunduh versi terbaru dataset House Prices
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    
    # Dataset biasanya berisi beberapa file, kita cari file CSV-nya
    downloaded_file = os.path.join(path, "Housing.csv")
    
    # Pastikan folder tujuan ada
    os.makedirs('data', exist_ok=True)
    target_path = 'data/harga_rumah.csv'
    
    # Copy file ke folder proyek kita
    shutil.copy(downloaded_file, target_path)
    print(f"Dataset berhasil disimpan di {target_path}")

    # Intip data sedikit
    df = pd.read_csv(target_path)
    print("\nPreview Data:")
    print(df[['area', 'price']].head())

if __name__ == "__main__":
    ingest_data()