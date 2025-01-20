import sqlite3

def setup_database():
    # Koneksi ke database SQLite
    conn = sqlite3.connect('hasil_deteksi.db')

    # Buat tabel jika belum ada
    conn.execute('''
    CREATE TABLE IF NOT EXISTS hasil_deteksi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT,
        car_count INTEGER,
        bike_count INTEGER,
        total_count INTEGER
    )
    ''')
    conn.commit()
    conn.close()

    print("Database dan tabel berhasil dibuat.")

if __name__ == '__main__':
    setup_database()
