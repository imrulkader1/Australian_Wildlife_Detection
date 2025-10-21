from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Live detection with all classes, high accuracy, and less misclassification
results = model.predict(
    source=0,      # Use webcam
    show=True,     # Display live detections
    conf=0.5,      # Higher confidence threshold to filter weak predictions
    iou=0.35,      # NMS IoU threshold to reduce overlapping boxes
    save=False     # Don't save outputs unless you want to
)

# Optional: Only print strong, high-confidence results
for r in results:
    for box in r.boxes:
        if box.conf > 0.6:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            print(f"Detected: {class_name}, Confidence: {box.conf:.2f}")
        else:
            print("Ignored low confidence prediction")
