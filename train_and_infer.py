import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
# CONFIG
# ---------------------------
xml_folder = "annotations_xml"        # folder with XML annotations
image_folder = "images_raw"           # folder with your raw images
dataset_folder = "dataset"            # YOLO dataset output
debug_folder = "debug_boxes"          # folder to save debug images

BOX_WIDTH = 40   # fallback fixed size
BOX_HEIGHT = 40

class_map = {
    "fish": 0,
    "1": 1
}

# Dataset structure
images_out = os.path.join(dataset_folder, "images")
labels_out = os.path.join(dataset_folder, "labels")
for split in ["train", "val"]:
    os.makedirs(os.path.join(images_out, split), exist_ok=True)
    os.makedirs(os.path.join(labels_out, split), exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# ---------------------------
# Helper: Expand point → bounding box
# ---------------------------
def expand_point_to_box(img, x, y, max_expand=50):
    """
    Try to expand from point (x,y) based on pixel differences.
    Falls back to fixed-size box if expansion fails.
    """
    h, w = img.shape[:2]
    thresh = 20  # pixel difference threshold
    visited = set()
    queue = [(x, y)]
    xmin, xmax, ymin, ymax = x, x, y, y

    try:
        base_color = img[y, x].astype(int)
    except:
        return x - BOX_WIDTH // 2, y - BOX_HEIGHT // 2, x + BOX_WIDTH // 2, y + BOX_HEIGHT // 2

    while queue and len(visited) < 500:
        cx, cy = queue.pop(0)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))

        if 0 <= cx < w and 0 <= cy < h:
            color = img[cy, cx].astype(int)
            if np.linalg.norm(color - base_color) < thresh:
                xmin, xmax = min(xmin, cx), max(xmax, cx)
                ymin, ymax = min(ymin, cy), max(ymax, cy)
                # Expand neighborhood
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = cx+dx, cy+dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))

    # If too small, fallback
    if (xmax - xmin) < 5 or (ymax - ymin) < 5:
        xmin = max(0, x - BOX_WIDTH // 2)
        ymin = max(0, y - BOX_HEIGHT // 2)
        xmax = min(w, x + BOX_WIDTH // 2)
        ymax = min(h, y + BOX_HEIGHT // 2)

    return xmin, ymin, xmax, ymax

# ---------------------------
# MAIN LOOP
# ---------------------------
xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]

for idx, xml_file in enumerate(xml_files):
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    base_name, _ = os.path.splitext(xml_file)
    filename = base_name + ".jpg"
    src_img_path = os.path.join(image_folder, filename)

    if not os.path.exists(src_img_path):
        print(f"⚠️ Image not found for {filename}, skipping...")
        continue

    img = cv2.imread(src_img_path)
    img_h, img_w = img.shape[:2]

    split = "train" if idx < int(0.8 * len(xml_files)) else "val"
    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in class_map:
            continue
        class_id = class_map[class_name]

        point = obj.find("point")
        bndbox = obj.find("bndbox")

        if point is not None:
            x = int(point.find("x").text)
            y = int(point.find("y").text)
            xmin, ymin, xmax, ymax = expand_point_to_box(img, x, y)

        elif bndbox is not None:
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

        else:
            continue

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Draw debug box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(img, class_name, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Save YOLO label
    if yolo_lines:
        txt_path = os.path.join(labels_out, split, base_name + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

    # Copy image to dataset
    dst_img_path = os.path.join(images_out, split, filename)
    shutil.copy(src_img_path, dst_img_path)

    # Save debug preview
    cv2.imwrite(os.path.join(debug_folder, filename), img)

    print(f"✅ Processed {filename} -> {split}")

print("✅ Conversion finished! Dataset + debug boxes ready.")

# ---------------------------
# TRAIN & INFERENCE
# ---------------------------
def train_yolo():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16
    )
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print("✅ Training finished! Best model at:", best_model_path)
    return best_model_path

def run_inference(model_path, test_folder="dataset/images/val", output_folder="predictions"):
    model = YOLO(model_path)
    results = model.predict(
        source=test_folder,
        save=True,
        save_txt=False,
        project=output_folder,
        name="predict"
    )
    print("✅ Predictions saved in:", results[0].save_dir)

# Example usage:
best_model = train_yolo()
run_inference(best_model)
