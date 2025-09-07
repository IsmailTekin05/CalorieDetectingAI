import json
import os
from pathlib import Path
from collections import defaultdict
import yaml

# ------------------------
# Base project path
# ------------------------
BASE_DIR = Path(__file__).resolve().parent

# ------------------------
# Paths (relative)
# ------------------------
train_json = BASE_DIR / "asset" / "public_training_set_release_2.0" / "annotations.json"
val_json   = BASE_DIR / "asset" / "public_validation_set_2.0" / "annotations.json"
train_img_folder = BASE_DIR / "asset" / "public_training_set_release_2.0" / "images"
val_img_folder   = BASE_DIR / "asset" / "public_validation_set_2.0" / "images"
output_labels_dir = BASE_DIR / "asset" / "labels"

os.makedirs(output_labels_dir / "train", exist_ok=True)
os.makedirs(output_labels_dir / "val", exist_ok=True)

# ------------------------
# Helper function
# ------------------------
def coco_to_yolo(json_file, img_folder, labels_out_dir, cat2id):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build mapping: image_id -> filename
    img_id2file = {img["id"]: img["file_name"] for img in data["images"]}
    img_id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    # Process each image
    for img_id, anns in annotations_by_image.items():
        img_file = img_id2file[img_id]
        img_path = Path(img_folder) / Path(img_file)
        if not img_path.exists():
            print(f"WARNING: {img_path} does not exist, skipping")
            continue

        w, h = img_id2size[img_id]
        txt_file = Path(labels_out_dir) / Path(img_file).with_suffix(".txt")
        txt_file.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            yolo_id = cat2id[cat_id]  # remap to 0-based
            bbox = ann["bbox"]  # COCO bbox: [x_min, y_min, width, height]
            x_c = (bbox[0] + bbox[2] / 2) / w
            y_c = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            lines.append(f"{yolo_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        with open(txt_file, "w") as f:
            f.write("\n".join(lines))

# ------------------------
# Prepare category mapping
# ------------------------
with open(train_json, "r", encoding="utf-8") as f:
    train_data = json.load(f)

categories = sorted(train_data["categories"], key=lambda x: x["id"])
cat2id = {cat["id"]: i for i, cat in enumerate(categories)}
names = {i: cat["name"] for i, cat in enumerate(categories)}

# ------------------------
# Convert datasets
# ------------------------
print("Converting training set...")
coco_to_yolo(train_json, train_img_folder, output_labels_dir / "train", cat2id)

print("Converting validation set...")
coco_to_yolo(val_json, val_img_folder, output_labels_dir / "val", cat2id)

# ------------------------
# Generate dataset.yaml
# ------------------------
dataset_yaml = {
    "path": str(output_labels_dir.resolve()),
    "train": "train",
    "val": "val",
    "nc": len(names),
    "names": names
}

with open(BASE_DIR / "dataset.yaml", "w") as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print("Conversion complete! dataset.yaml created. Ready for YOLOv8 training.")
