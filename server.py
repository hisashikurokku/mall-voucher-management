import requests
from datetime import datetime

BACKEND_URL = "http://localhost:8000/api/events/plate-detected"

def send_plate_event(plate_text, camera_id="entrance_1", confidence=None):
    payload = {
        "plate_number": plate_text,
        "camera_id": camera_id,
        "detected_at": datetime.utcnow().isoformat(),
        "confidence": confidence
    }
    try:
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        print("Event sent to backend:", response.json())
    except Exception as e:
        print("Error sending event:", e)
