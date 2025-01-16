from twilio.rest import Client

def send_whatsapp_notification(vehicle_count):
    account_sid = 'AC9e351ef437cb18718fc924cc5aec6b1d'
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