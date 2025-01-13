import os
import cv2
import math
from time import sleep
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.utils import secure_filename
from flask import jsonify
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'your_secret_key'

# Load Cascade Classifiers
car_cascade = cv2.CascadeClassifier('models/cascade.xml')
bike_cascade = cv2.CascadeClassifier('models/motor2.xml')

car_count, bike_count = 0, 0  # Variabel untuk counting kendaraan

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Tidak dapat membuka video"
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return duration

# Fungsi untuk membuat Kalman Filter baru
def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)
    return kf

# Fungsi untuk memeriksa overlap
def check_overlap(rect1, rect2, threshold=0.5):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Hitung area overlap
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return False  # Tidak ada overlap

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2

    # Menghitung IoU (Intersection over Union)
    overlap_ratio = overlap_area / float(area1 + area2 - overlap_area)
    return overlap_ratio > threshold

@app.route('/vehicle_count')
def vehicle_count():
    return jsonify(car_count=car_count, bike_count=bike_count)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    global car_count, bike_count
    car_count, bike_count = 0, 0
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('Tidak ada file video!')
            return redirect(url_for('upload_video'))
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload_video.html', filename=filename)
        else:
            flash('Format file tidak didukung!')
            return redirect(url_for('upload_video'))
    return render_template('upload_video.html')

@app.route('/detect_stream/<filename>')
def detect_stream(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(video_path):
        return "File tidak ditemukan", 404
    return Response(detect_and_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Fungsi untuk mendeteksi dan melacak kendaraan
def detect_and_stream(video_path):
    global car_count, bike_count
    car_count, bike_count = 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka video.")
        return

    roi_top, roi_bottom = 200, 550
    pos_line = 120
    offset = 5
    cars_kf = []  # Daftar Kalman Filter untuk mobil
    bikes_kf = []  # Daftar Kalman Filter untuk motor

    def center_object(x, y, w, h):
        return np.array([[x + w // 2], [y + h // 2]], np.float32)

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            break

        # Gambarkan ROI
        cv2.rectangle(img, (0, roi_top), (img.shape[1], roi_bottom), (0, 255, 0), 2)
        cv2.line(img, (0, roi_top), (img.shape[1], roi_top), (255, 0, 0), 2)  # Garis atas ROI
        cv2.line(img, (0, roi_bottom), (img.shape[1], roi_bottom), (255, 0, 0), 2)  # Garis bawah ROI

        img_roi = img[roi_top:roi_bottom, :]
        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

        # Deteksi mobil dan motor
        cars = car_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=12, minSize=(100, 100))
        bikes = bike_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=13, minSize=(20, 20))

        # Update Kalman Filter untuk setiap deteksi mobil
        for (x, y, w, h) in cars:
            measurement = center_object(x, y, w, h)
            matched = False

            # Gambarkan bounding box untuk mobil
            cv2.rectangle(img, (x, y + roi_top), (x + w, y + h + roi_top), (0, 0, 255), 2)
            cv2.putText(img, "Mobil", (x, y + roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for kf in cars_kf:
                prediction = kf.predict()
                if np.linalg.norm(prediction[:2] - measurement[:2]) < 50:
                    kf.correct(measurement)
                    matched = True
                    break

            if not matched:
                kf = create_kalman_filter()
                kf.statePre = np.array([[x + w // 2], [y + h // 2], [0], [0]], np.float32)
                kf.statePost = kf.statePre.copy()
                cars_kf.append(kf)
                car_count += 1

        # Update Kalman Filter untuk setiap deteksi motor
        for (x, y, w, h) in bikes:
            measurement = center_object(x, y, w, h)
            matched = False

            # Gambarkan bounding box untuk motor
            cv2.rectangle(img, (x, y + roi_top), (x + w, y + h + roi_top), (255, 0, 0), 2)
            cv2.putText(img, "Motor", (x, y + roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for kf in bikes_kf:
                prediction = kf.predict()
                if np.linalg.norm(prediction[:2] - measurement[:2]) < 50:
                    kf.correct(measurement)
                    matched = True
                    break

            if not matched:
                kf = create_kalman_filter()
                kf.statePre = np.array([[x + w // 2], [y + h // 2], [0], [0]], np.float32)
                kf.statePost = kf.statePre.copy()
                bikes_kf.append(kf)
                bike_count += 1

        # Prediksi posisi untuk objek yang tidak dideteksi
        for kf in cars_kf + bikes_kf:
            kf.predict()

        # Visualisasi hasil deteksi
        cv2.putText(img, f"Mobil: {car_count}", (450, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Motor: {bike_count}", (650, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode frame untuk streaming
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/hasil_deteksi/<filename>')
def hasil_deteksi(filename):
    global car_count, bike_count
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Data untuk frontend
    data = {
        'car_count': car_count,
        'bike_count': bike_count,
        'video_name': filename
    }
    return render_template('hasil_deteksi.html', data=data)

@app.route('/hasil_pengujian', methods=['GET', 'POST'])
def hasil_pengujian():
        
    filename = request.args.get('filename', 'No file selected')
    car_count = int(request.args.get('car_count', 0))
    bike_count = int(request.args.get('bike_count', 0))
    
    data = {
        'car_count': car_count,
        'bike_count': bike_count,
        'video_name': filename,
        'evaluation': None,
        'manual_car_count': None,
        'manual_bike_count': None
    }

    if request.method == 'POST':
        # Mengambil input jumlah manual
        manual_car_count = int(request.form.get('manual_car_count', 0))
        manual_bike_count = int(request.form.get('manual_bike_count', 0))
        
        # Membandingkan hasil deteksi dengan input manual
        true_labels_car = [1] * manual_car_count + [0] * (car_count - manual_car_count)
        predicted_labels_car = [1] * car_count + [0] * (manual_car_count - car_count)

        true_labels_bike = [1] * manual_bike_count + [0] * (bike_count - manual_bike_count)
        predicted_labels_bike = [1] * bike_count + [0] * (manual_bike_count - bike_count)
        
        # Menghitung metrik evaluasi
        evaluation = {
            'accuracy_car': accuracy_score(true_labels_car, predicted_labels_car) * 100,
            'precision_car': precision_score(true_labels_car, predicted_labels_car, zero_division=0),
            'recall_car': recall_score(true_labels_car, predicted_labels_car, zero_division=0),
            'f1_car': f1_score(true_labels_car, predicted_labels_car, zero_division=0),
            'accuracy_bike': accuracy_score(true_labels_bike, predicted_labels_bike) * 100,
            'precision_bike': precision_score(true_labels_bike, predicted_labels_bike, zero_division=0),
            'recall_bike': recall_score(true_labels_bike, predicted_labels_bike, zero_division=0),
            'f1_bike': f1_score(true_labels_bike, predicted_labels_bike, zero_division=0)
        }

        # Update data untuk diteruskan ke template
        data.update({
            'manual_car_count': manual_car_count,
            'manual_bike_count': manual_bike_count,
            'evaluation': evaluation
        })

    return render_template('hasil_pengujian.html', filename=filename, data=data)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
