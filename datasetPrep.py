# ---------------------------------------------
# Convert COCO Annotations to YOLO and Organize Dataset
# ---------------------------------------------
"""
Dataset citation:
Qianqian Zhang and Khandakar Amed. Australia Animal Species Image Dataset (50). Kaggle. 2025.
DOI: 10.34740/KAGGLE/DSV/12990738
URL: https://www.kaggle.com/datasets/entenam/australia-animal-species-image-dataset-47/
License: CC BY-NC 4.0 (Attribution-NonCommercial)
"""
# -------------------------------
# 1: Import libraries
# -------------------------------
import os
import json
import shutil
import random
import cv2

# -------------------------------
# 2: Set up path variables and folders
# -------------------------------
# Folder "dataset" contains all species folders
root_dir = 'dataset'
out_dir = 'dataset_yolo'

# Create output folders for train/val images and labels
splits = ['train', 'val']
for split in splits:
    os.makedirs(os.path.join(out_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, split, 'labels'), exist_ok=True)

# -------------------------------
# 3: Gather all species folders for processing
# -------------------------------
species_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f[0].isdigit()]

# -------------------------------
# 4: Split data, convert, and organize
# -------------------------------
val_split = 0.2  # 20% for validation

for species in species_folders:
    img_folder = os.path.join(root_dir, species, "images with background")
    annot_folder = os.path.join(root_dir, species, "annotations_coco")
    images = sorted([f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')])

    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_split))
    split_data = [("train", images[:split_idx]), ("val", images[split_idx:])]

    for split, imgs in split_data:
        for img_file in imgs:
            img_stem = os.path.splitext(img_file)[0]
            json_file = os.path.join(annot_folder, f"{img_stem}_coco.json")
            out_img = os.path.join(out_dir, split, "images", img_file)
            out_label = os.path.join(out_dir, split, "labels", img_stem + '.txt')

            # Copy image
            shutil.copy2(os.path.join(img_folder, img_file), out_img)

            # Convert annotation
            if os.path.isfile(json_file):
                with open(json_file, 'r') as jf:
                    data = json.load(jf)
                if isinstance(data, list):
                    data = data[0]
                bbox = data['bbox']  # COCO: x_min, y_min, width, height
                category_id = data['category_id'] - 1  # YOLO category index starts from 0
                img_path = os.path.join(img_folder, img_file)
                img_cv = cv2.imread(img_path)
                height, width = img_cv.shape[:2]
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height
                yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                with open(out_label, 'w') as lf:
                    lf.write(yolo_line)

# -------------------------------
# 5: Results
# -------------------------------
print("Conversion and organization complete.")
print("Ready-to-train dataset in folder: dataset_yolo")

# -------------------------------
# End of Pipeline
# -------------------------------