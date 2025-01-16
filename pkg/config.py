import os
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()

class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')  # default ke localhost jika tidak ada
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')      # default ke 'root' jika tidak ada
    MYSQL_DB = os.getenv('MYSQL_NAME', 'nama_database')  # default ke 'nama_database' jika tidak ada

    # Konfigurasi SQLAlchemy untuk Flask-Migrate
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:@{MYSQL_HOST}/{MYSQL_DB}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Menonaktifkan notifikasi perubahan