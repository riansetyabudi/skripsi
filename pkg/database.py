import pymysql
from pymysql.err import OperationalError, InternalError
from pkg.config import Config

# Variabel global untuk menyimpan koneksi
db_connection = None

def init_db_connection():
    """Inisialisasi koneksi database saat aplikasi dimulai."""
    global db_connection
    try:
        if db_connection is None:
            db_connection = pymysql.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,  # Pastikan menambahkan password
                database=Config.MYSQL_DB,
                cursorclass=pymysql.cursors.DictCursor,
            )
            print("Sukses konek ke database cuyy")
        else:
            print("Database sudah terkoneksi")
    except OperationalError as oe:
        print(f"Gagal koneksi ke database: {oe}")
        db_connection = None
    except InternalError as ie:
        print(f"Masalah internal database: {ie}")
        db_connection = None
    except Exception as e:
        print(f"Error tidak terduga: {e}")
        db_connection = None

def get_db_connection():
    """Mengembalikan koneksi database global."""
    global db_connection
    if db_connection is None:
        raise RuntimeError("Database connection is not initialized. Call init_db_connection() first.")
    return db_connection

def close_db_connection():
    """Menutup koneksi database saat aplikasi dihentikan."""
    global db_connection
    if db_connection is not None:
        db_connection.close()
        db_connection = None