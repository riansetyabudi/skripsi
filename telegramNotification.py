import requests

TELEGRAM_TOKEN = "7697149692:AAGzIlzG-lavQycHeRwt_YKz9Y1KUAJUxp4"  # Ganti dengan token bot Anda
TELEGRAM_CHAT_ID = "6266299771"  # Ganti dengan chat_id Anda

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("Pesan berhasil dikirim ke Telegram")
    else:
        print(f"Error mengirim pesan: {response.text}")
