#convert_yolo _to_cnn_test.py
import os
import shutil
import glob

# === PATHS ===
yolo_test_images = "movement detection.v1i.yolov8/test/images"
yolo_test_labels = "movement detection.v1i.yolov8/test/labels"
cnn_test_dir = "cnn_test_data"

movement_dir = os.path.join(cnn_test_dir, "movement_detected")
no_movement_dir = os.path.join(cnn_test_dir, "no_movement")

# === CREATE DESTINATION FOLDERS ===
os.makedirs(movement_dir, exist_ok=True)
os.makedirs(no_movement_dir, exist_ok=True)

# === CONVERT LABELS TO CLASSIFICATION FOLDERS ===
for label_file in os.listdir(yolo_test_labels):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(yolo_test_labels, label_file)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Check if movement class (usually 1)
    has_movement = any(line.startswith("1") for line in lines)

    # Match label file to corresponding image
    base_filename = os.path.splitext(label_file)[0]
    matching_images = glob.glob(os.path.join(yolo_test_images, f"{base_filename}.jpg.rf.*.jpg"))

    if matching_images:
        img_path = matching_images[0]
        img_file = os.path.basename(img_path)
        dest_folder = movement_dir if has_movement else no_movement_dir
        shutil.copy(img_path, os.path.join(dest_folder, img_file))
    else:
        print(f"❌ No image found for {label_file}")

print("✅ Conversion complete: Test images are now in 'cnn_test_data/' folder.")
