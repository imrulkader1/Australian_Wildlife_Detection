#---------------------------------------------
# Edge AI Wildlife Detection & Telemetry
#---------------------------------------------

#---------------------------------------------
# 1. Live Detection, Logging, Transmission
#---------------------------------------------
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO

#---------------------------------------------
# 2. Model Initialization
#---------------------------------------------
model = YOLO("runs/detect/train/weights/best.pt")

#---------------------------------------------
# 3. Location Setup
#---------------------------------------------
def read_location(filename="pi_location.txt"):
    with open(filename, "r") as f:
        lat, lon = f.read().strip().split(",")
        return float(lat), float(lon)

DEVICE_LAT, DEVICE_LON = read_location()
DEVICE_LOCATION = {"latitude": DEVICE_LAT, "longitude": DEVICE_LON}

#---------------------------------------------
# 4. Local Storage Setup
#---------------------------------------------
LOG_PATH = "detections_log.csv"
header = ["datetime", "class_name", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "location_lat", "location_lon"]

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

#---------------------------------------------
# 5. Log Detections Function
#---------------------------------------------
def log_detection(class_name, confidence, bbox, location):
    now = datetime.utcnow().isoformat()
    conf_float = float(confidence)
    row = [
        now, class_name, f"{conf_float:.2f}",
        bbox[0], bbox[1], bbox[2], bbox[3],
        location["latitude"], location["longitude"]
    ]
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

#---------------------------------------------
# 6. Wi-Fi Check & Telemetry Placeholders
#---------------------------------------------
def check_wifi():
    # TODO: Implement system/network dependent Wi-Fi connectivity check
    return False

def transmit_to_server(file_path):
    # TODO: Implement server/cloud upload here using requests, MQTT, FTP, etc.
    print(f"Transmitting {file_path} to central server...")

#---------------------------------------------
# 7. Detection Duration & Cooldown Logic
#---------------------------------------------
detection_times = {}
last_logged = {}
DETECTION_THRESHOLD = 3  # seconds continous before logging
COOLDOWN_PERIOD = 5      # seconds after logging

#---------------------------------------------
# 8. Real-Time Detection & Logging Loop
#---------------------------------------------
print("Starting real-time detection...")
while True:
    DEVICE_LAT, DEVICE_LON = read_location()
    DEVICE_LOCATION = {"latitude": DEVICE_LAT, "longitude": DEVICE_LON}
    now_ts = time.time()
    results = model.predict(
        source=0,
        show=True,
        conf=0.5,
        iou=0.35,
        stream=True,
        save=False
    )
    for r in results:
        for box in r.boxes:
            conf_float = float(box.conf)
            if conf_float > 0.6:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                bbox = box.xywh[0].tolist() if hasattr(box.xywh[0], "tolist") else list(box.xywh[0])
                # Track detection time
                if class_name not in detection_times:
                    detection_times[class_name] = now_ts
                # Only log if present > DETECTION_THRESHOLD seconds and cooldown is satisfied
                if (now_ts - detection_times[class_name] >= DETECTION_THRESHOLD and
                    (class_name not in last_logged or now_ts - last_logged[class_name] > COOLDOWN_PERIOD)):
                    log_detection(class_name, conf_float, bbox, DEVICE_LOCATION)
                    print(f"{datetime.now()} - {class_name} DETECTED & LOGGED")
                    last_logged[class_name] = now_ts
                    detection_times[class_name] = now_ts  # Reset detection to avoid continuous logging
                else:
                    print(f"{datetime.now()} - {class_name} detected (waiting for threshold/cooldown...)")
            else:
                print("Low-confidence detection ignored")
    if check_wifi():
        transmit_to_server(LOG_PATH)
    time.sleep(1)

#---------------------------------------------
# End of Pipeline
#---------------------------------------------
