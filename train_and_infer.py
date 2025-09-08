import os
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import cv2
import os


# CONFIG
# ---------------------------
xml_folder = "annotations_xml"        # folder containing your XML files
image_folder = "images_raw"           # folder where your original images are stored
dataset_folder = "dataset"            # final YOLO dataset folder

# Dataset structure
images_out = os.path.join(dataset_folder, "images")
labels_out = os.path.join(dataset_folder, "labels")
for split in ["train", "val"]:
    os.makedirs(os.path.join(images_out, split), exist_ok=True)
    os.makedirs(os.path.join(labels_out, split), exist_ok=True)

# Fixed bounding box size (in pixels) - adjust based on fish size in dataset
BOX_WIDTH = 40
BOX_HEIGHT = 40

# Class mapping
class_map = {
    "fish": 0,
    "1": 1   # example if class '1' is valid
}

# ---------------------------
# MAIN LOOP
# ---------------------------
xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]

for idx, xml_file in enumerate(xml_files):
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    # Get image info
    filename = root.find("filename").text
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    # Choose split (simple: 80% train, 20% val)
    split = "train" if idx < int(0.8 * len(xml_files)) else "val"

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in class_map:
            continue
        class_id = class_map[class_name]

        # Get point
        point = obj.find("point")
        x = int(point.find("x").text)
        y = int(point.find("y").text)

        # Convert point → bbox
        xmin = max(0, x - BOX_WIDTH // 2)
        ymin = max(0, y - BOX_HEIGHT // 2)
        xmax = min(img_w, x + BOX_WIDTH // 2)
        ymax = min(img_h, y + BOX_HEIGHT // 2)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Save label file
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(labels_out, split, base_name + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # Copy corresponding image into images/ folder
    src_img_path = os.path.join(image_folder, filename + ".jpg")  # adjust extension if needed
    dst_img_path = os.path.join(images_out, split, filename + ".jpg")
    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dst_img_path)

print("✅ Conversion finished! YOLO dataset ready in:", dataset_folder)

# Code: Train + Inference code 



# 1. Train the model
def train_yolo():
    # Choose a small model (yolov8n) for speed, or use yolov8s / yolov8m for more accuracy
    model = YOLO("yolov8n.pt")  

    # Train with your dataset
    model.train(
        data="dataset/data.yaml",  # path to your YAML file
        epochs=50,
        imgsz=640,
        batch=16
    )

    # Save best model path
    return "runs/detect/train/weights/best.pt"



def run_inference(model_path, test_folder="dataset/images/val", output_folder="predictions"):
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(model_path)

    # Run prediction
    results = model.predict(source=test_folder, save=True, save_txt=True, project=output_folder)

    print("✅ Predictions saved in:", results[0].save_dir)

    # OPTIONAL: display the images with boxes
    for r in results:
        img_with_boxes = r.plot()   # numpy array with bounding boxes
        cv2.imshow("Detections", img_with_boxes)
        cv2.waitKey(0)              # press any key for next image
    cv2.destroyAllWindows()

# Training
best_model = train_yolo()

# inference
run_inference(best_model)