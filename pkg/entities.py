from datetime import datetime
import pytz
from app import db 


class Detection(db.Model):
    __tablename__ = 'detections'  # Nama tabel di database
    id = db.Column(db.Integer, primary_key=True)  # Kolom id
    video_name = db.Column(db.String(255))  # Kolom video_name
    car_amount = db.Column(db.Integer)  # Kolom vehicle_type
    bike_amount = db.Column(db.Integer)  # Kolom amount
    total_amount = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now(pytz.timezone("Asia/Jakarta")))  # Kolom created_at

    def __init__(self, video_name, car_amount, bike_amount, total_amount):
        self.video_name = video_name
        self.car_amount = car_amount
        self.bike_amount = bike_amount
        self.total_amount = total_amount

    def to_dict(self):
        """Convert object to dictionary."""
        return {
            "id": self.id,
            "video_name": self.video_name,
            "car_amount": self.car_amount,
            "bike_amount": self.bike_amount,
            "total_amount": self.total_amount,
            "created_at": self.created_at,
        }

class ComparisonResult(db.Model):
    __tablename__ = 'comparison_results'  # Nama tabel di database
    comparison_id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('detections.id'))  # Hubungkan dengan tabel detections
    system_calculation_result = db.Column(db.Float)
    manual_calculation_result = db.Column(db.Float)

    def __init__(self, detection_id, system_calculation_result, manual_calculation_result):
        self.detection_id = detection_id
        self.system_calculation_result = system_calculation_result
        self.manual_calculation_result = manual_calculation_result

    def to_dict(self):
        return {
            "comparison_id": self.comparison_id,
            "detection_id": self.detection_id,
            "system_calculation_result": self.system_calculation_result,
            "manual_calculation_result": self.manual_calculation_result,
        }

class MetrixEvaluation(db.Model):
    __tablename__ = 'metrix_evaluations'  # Nama tabel di database
    comparison_id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)

    def __init__(self, accuracy, precision, recall, f1_score):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def to_dict(self):
        return {
            "comparison_id": self.comparison_id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
}