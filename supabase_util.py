import os
from flask import Flask, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(url, key)

# Fungsi untuk upload video ke Supabase Storage dengan cek keberadaan file
def upload_to_supabase(file, storage_bucket, destination_path):
    print(f"Starting upload for {destination_path}...")

    try:
        # Cek apakah file sudah ada di storage
        existing_files = supabase.storage.from_(storage_bucket).list(
            path=destination_path.rsplit('/', 1)[0]  # Mendapatkan direktori tujuan
        )
        print(f"files : {existing_files}")

        if any(file['name'] == destination_path.rsplit('/', 1)[-1] for file in existing_files):
            print(f"File {destination_path} sudah ada. File akan ditimpa.")

            # Hapus file lama sebelum upload (opsional, tergantung preferensi)
            delete_response = supabase.storage.from_(storage_bucket).remove([destination_path])
            if hasattr(delete_response, 'error'):
                print(f"Error saat menghapus file lama: {delete_response['error']}")
                return {"error": delete_response['error']}

        # Unggah file baru (atau timpa file lama)
        response = supabase.storage.from_(storage_bucket).upload(
            path=destination_path,
            file=file,
            file_options={"cache-control": "3600"}
        )

        print(f"Upload to {destination_path} complete!")
        
        # Cek apakah respons upload berhasil atau ada error
        if hasattr(response, 'error') and response['error']:
            print(f"Error during upload: {response['error']}")
            return {"error": response["error"]}

        return {"success": True, "destination_path": destination_path}

    except Exception as e:
        print(f"Exception during upload: {str(e)}")
        return {"error": str(e)}

def get_public_url(storage_bucket, destination_path):
    try:
        # Mendapatkan URL publik untuk file yang diunggah
        public_url_response = supabase.storage.from_(storage_bucket).get_public_url(destination_path)
        print(f"public url : {public_url_response}")
        # Cek jika public_url_response ada dan memiliki URL yang valid
        if public_url_response != "":
            public_url = public_url_response
            if public_url:
                print(f"File URL: {public_url}")
                return {"success": True, "file_url": public_url}
            else:
                print("URL kosong, periksa pengaturan bucket di Supabase!")
                return {"error": "URL kosong"}
        else:
            print("Error getting public URL")
            return {"error": "Error getting public URL"}
    except Exception as e:
        print(f"Exception during URL fetch: {str(e)}")
        return {"error": str(e)}

def upload_video_file(file, storage_bucket, destination_path):
    upload_response = upload_to_supabase(file, storage_bucket, destination_path)
    
    if "success" in upload_response:
        # Jika upload berhasil, dapatkan public URL
        file_url_response = get_public_url(storage_bucket, destination_path)
        print("berhasil upload")
        
        if "success" in file_url_response:
            print("berhasil dapetin url")
            # Kembalikan URL file jika berhasil
            return {"success": True, "file_url": file_url_response["file_url"]}
        else:
            return {"error": file_url_response["error"]}

    else:
        return {"error": upload_response["error"]}