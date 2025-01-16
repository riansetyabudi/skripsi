import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

def send_whatsapp_notification(vehicle_count):
    account_sid = os.getenv('TWILIO_SID') 
    auth_token = '[AuthToken]'  # Ganti dengan Auth Token Anda
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=f"Peringatan: Lalu lintas tinggi! Total kendaraan: {vehicle_count}.",
            to='whatsapp:+6281249304189'
        )
        print(f"Pesan terkirim dengan SID: {message.sid}")
    except Exception as e:
        print(f"Error: {e}")