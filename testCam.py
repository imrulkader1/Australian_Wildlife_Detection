#---------------------------------------------
# Edge AI Wildlife Detection & Telemetry
#---------------------------------------------

#---------------------------------------------
# 1. Live Detection, Logging, Transmission
#---------------------------------------------
import os
import csv
import time
import requests
from datetime import datetime, timezone
from ultralytics import YOLO
from github import Github

#---------------------------------------------
# 2. Model Initialization
#---------------------------------------------
model = YOLO("best.pt")

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
    now = datetime.now(timezone.utc).isoformat()
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
# 6. Wi-Fi Check & Telemetry
#---------------------------------------------
#TOKEN = "REMOVED(PERSONAL TOKEN)" #GitHub Token
REPO_NAME = "wildlife-edge-ai-logs"
FILE_PATH = LOG_PATH
TARGET_PATH = "detections_log.csv"
COMMIT_MSG = "Auto-upload wildlife detection log"

def upload_to_github(token, repo_name, file_path, target_path, commit_msg="Auto-upload"):
    g = Github(token)
    user = g.get_user()
    try:
        repo = user.get_repo(repo_name)
        print(f"Found repo '{repo_name}'.")
    except Exception as e:
        print(f"Repository '{repo_name}' not found: {e}")
        try:
            repo = user.create_repo(repo_name)
            print(f"Repository '{repo_name}' created.")
        except Exception as e2:
            print(f"Failed to create repo: {e2}")
            return
    try:
        with open(file_path, "rb") as file_handle:
            content = file_handle.read()
    except Exception as e:
        print(f"Could not read file '{file_path}': {e}")
        return
    try:
        file = repo.get_contents(target_path)
        repo.update_file(target_path, commit_msg, content, file.sha)
        print(f"Updated '{target_path}' on GitHub.")
    except Exception as e:
        print(f"Update failed: {e}")
        try:
            repo.create_file(target_path, commit_msg, content)
            print(f"Created '{target_path}' on GitHub.")
        except Exception as e2:
            print(f"Create failed: {e2}")

def check_wifi():
    try:
        requests.get("https://github.com", timeout=3)
        return True
    except requests.RequestException as e:
        print(f"Internet check failed: {e}")
        return False

def transmit_to_server(file_path):
    upload_to_github(TOKEN, REPO_NAME, file_path, TARGET_PATH, COMMIT_MSG)

#---------------------------------------------
# 7. Detection Duration & Cooldown Logic
#---------------------------------------------
detection_times = {}
last_logged = {}
DETECTION_THRESHOLD = 0  # seconds continuous before logging
COOLDOWN_PERIOD = 1      # seconds after logging

#---------------------------------------------
# 8. Detection Filtering Option
#---------------------------------------------
# List of species
# 0: Dromaius novaehollandiae (Emu)
# 1: Macropus giganteus (Kangaroo)
# 2: Phascolarctos cinereus (Koala)
# 36: Vombatus ursinus (Wombat)
# To detect only these, FILTER_CLASSES = [0, 1, 2, 36]
# For no filter, FILTER_CLASSES = None
FILTER_CLASSES = [0, 1, 2, 5, 12, 14, 15, 20, 23, 24, 25, 26, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 44, 45]

#---------------------------------------------
# 9. Real-Time Detection & Logging Loop
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
        save=False,
        classes=FILTER_CLASSES  # Filter detection to specific indices
    )
    for r in results:
        for box in r.boxes:
            conf_float = float(box.conf)
            if conf_float > 0.5:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                bbox = box.xywh[0].tolist() if hasattr(box.xywh[0], "tolist") else list(box.xywh[0])
                # Track detection time
                if class_name not in detection_times:
                    detection_times[class_name] = now_ts
                if (now_ts - detection_times[class_name] >= DETECTION_THRESHOLD and
                    (class_name not in last_logged or now_ts - last_logged[class_name] > COOLDOWN_PERIOD)):
                    log_detection(class_name, conf_float, bbox, DEVICE_LOCATION)
                    print(f"{datetime.now()} - {class_name} DETECTED & LOGGED")
                    last_logged[class_name] = now_ts
                    detection_times[class_name] = now_ts
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
