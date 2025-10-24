# Australian Wildlife Detection – Edge AI Platform

A **autonomous wildlife monitoring system** for Australian national parks, powered by Edge Artificial Intelligence. Real-time animal detection and identification on a **Raspberry Pi 4**, bringing advanced machine learning directly to the wild.

The solution supports conservation and ecological research by automating data collection, reducing manual fieldwork, and working even without continuous internet connectivity.

---

## Key Features

- **Real-Time Animal Detection:**  
  Fine-tuned **YOLOv8-Nano** model recognizing **50+ Australian species** efficiently.

- **Configurable Detection:**  
  Expand or restrict target species via the `FILTER_CLASSES` variable.

- **Automated Logging:**  
  Logs: detection time, GPS coordinates, class, confidence, bounding box to `detections_log.csv`.

- **Edge Processing:**  
  Runs totally offline—inference on **low-power Raspberry Pi** hardware.

- **Offline-First, Cloud Sync:**  
  Auto-upload detection logs to GitHub whenever Wi-Fi is available using PyGithub.

- **Reproducible Training:**  
  Dataset curation, annotation conversion, YOLO training: all scripts included.

- **Modular Codebase:**  
  Ready for new species, datasets, or sites—adapt quickly with clear script separation.

---

## Installation & Setup

### Requirements

- Python **3.10+**
- Libraries: PyTorch, Ultralytics YOLO, OpenCV, etc. *(see `requirements.txt`)*
- Hardware: **Raspberry Pi 4** or compatible Linux system
- Camera: USB or Pi camera  
- GPS and PIR motion sensor for extra telemetry

> See requirements.txt for details.

### Quick Start

**Clone the repository**
 ```
git clone https://github.com/imrulkader1/Australian_Wildlife_Detection.git

cd Australian_Wildlife_Detection
 ```
**Install dependencies**
 ```
pip install -r requirements.txt
 ```
---

## Model Training

1. **Prepare the dataset:**
 ```
python datasetPrep.py
 ```
2. **Train YOLOv8Nano:**
 ```
python trainingModel.py
 ```
Or use Ultralytics CLI:
 ```
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
 ```

---

## Deploy and Run Detection

1. Edit detection preferences in `testCam.py`:
 ```
 FILTER_CLASSES = None  # Detect all species
 # Or example subset:
 FILTER_CLASSES =   # Use class indices from model[1][2]
 ```

2. Place `best.pt` (YOLO weights) in your working directory.

3. Run main detection loop:
python testCam.py

4. Detections logged locally to `detections_log.csv` and synced to GitHub on Wi-Fi.

---

## Dataset

- **Source:** Australia Animal Species Image Dataset (50)  
- **Citation:** Zhang, Q. & Amed, K. (2025). DOI: 10.34740/KAGGLE/DSV/12990738  
- Curated, annotated, and split using `datasetPrep.py`.

---

## Outputs

| File                              | Description                                                      |
|------------------------------------|------------------------------------------------------------------|
| detections_log.csv                 | Real-time detection log: datetime, class, confidence, location, bbox |
| results.csv                        | Test results and validation metrics                              |
| runs/detect/train/ (images/plots)  | Sample predictions, confusion matrix, batch visualizations       |

---

**Telemetry & Sync:**  
Detection logs auto-upload to GitHub when Wi-Fi is available; see `check_wifi()` and upload logic in `testCam.py`.

---

## File Descriptions

| File                  | Purpose                                                   |
|-----------------------|-----------------------------------------------------------|
| testCam.py            | Main detection, logging, and auto-upload to GitHub        |
| trainingModel.py      | YOLOv8Nano training and evaluation routines               |
| datasetPrep.py        | Dataset curation, splitting, and annotation conversion    |
| detections_log.csv    | Log of all detected events (local & upload)               |
| results.csv           | Model test/validation results                             |
| requirements.txt      | Requirements based on all project code                    |
| runs/detect/train/    | Sample predictions, confusion matrices, batch visualizations |

---

## References

- Zhang, Q. & Amed, K. (2025). *Australia Animal Species Image Dataset*. Kaggle. DOI: 10.34740/KAGGLE/DSV/12990738  
- Ultralytics YOLOv8 Documentation  
- Raspberry Pi Foundation – Camera Interface Guide  
- OpenCV Python Documentation  
- PyGithub documentation  

---

## License

- **Code:** MIT License  
- **Dataset:** CC BY-NC 4.0
