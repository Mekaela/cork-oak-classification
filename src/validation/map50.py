import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

IMAGE_DIR = Path("data/yolo_plots_data/images/val")
LABEL_DIR = Path("data/yolo_plots_data/labels/val")
OUT_JSON = "data/yolo_plots_data/tree_crowns_val_coco.json"

CLASS_NAME = "tree"
COCO_CATEGORY_ID = 1  # COCO IDs conventionally start at 1

def create_coco_json(image_dir, label_dir, out_json):
    images = []
    annotations = []
    categories = [{"id": COCO_CATEGORY_ID, "name": CLASS_NAME}]

    image_id = 0
    annotation_id = 0

    for image_path in sorted(IMAGE_DIR.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue

        image_id += 1

        with Image.open(image_path) as im:
            width, height = im.size

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = LABEL_DIR / (image_path.stem + ".txt")
        if not label_path.exists():
            continue  # valid: an image with no reference crowns

        for line in label_path.read_text().splitlines():
            values = line.strip().split()
            if len(values) < 7:  # class + at least 3 x,y pairs
                continue

            # values[0] is the YOLO class ID; discard it for a one-class dataset
            coords = list(map(float, values[1:]))

            # YOLO polygon coordinates are normalized:
            # x values are multiplied by image width,
            # y values by image height.
            polygon = []
            for i in range(0, len(coords), 2):
                polygon.extend([
                    coords[i] * width,
                    coords[i + 1] * height,
                ])

            xs = polygon[0::2]
            ys = polygon[1::2]

            # Shoelace formula for polygon area
            area = abs(sum(
                xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                for i in range(len(xs))
            )) / 2.0

            annotation_id += 1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": COCO_CATEGORY_ID,
                "segmentation": [polygon],  # one polygon: [[x1,y1,x2,y2,...]]
                "bbox": [
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                ],
                "area": area,
                "iscrowd": 0,
            })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(coco, f)

    print(f"Wrote {len(images)} images and {len(annotations)} crown annotations to {OUT_JSON}")


import json
def check_files():
    with open("data/yolo_plots_data/tree_crowns_val.json") as f:
        gt = json.load(f)

    with open("runs_yolo/runs_yolo8_plots_150epochs/segment/yolo_plots_run/predictions.json") as f:
        preds = json.load(f)

    gt_image_ids = {img["id"] for img in gt["images"]}
    pred_image_ids = {p["image_id"] for p in preds}

    print("GT image IDs:", sorted(gt_image_ids)[:10])
    print("Prediction image IDs:", sorted(pred_image_ids)[:10])
    print("Prediction IDs missing from GT:", pred_image_ids - gt_image_ids)




WEIGHTS = "runs_yolo/runs_yolo8_site_300epochs/segment/yolo_plots_run/weights/best.pt"
DATA_YAML = "models/training_plots/data.yaml"

model = YOLO(WEIGHTS)

metrics = model.val(
    data=DATA_YAML,
    split="val",       # use "val" if that is your evaluation split
    imgsz=640,          # use the same image size as training/validation
    conf=0.001,         # retain low-confidence predictions for AP
    iou=0.5,            # NMS IoU threshold, not the mAP IoU threshold
    save_json=True,
    save_conf=True,
    verbose=True,
)

print(f"Segmentation mAP50:    {metrics.seg.map50:.4f}")
print(f"Segmentation mAP75:    {metrics.seg.map75:.4f}")
print(f"Segmentation mAP50-95: {metrics.seg.map:.4f}")

print(f"Box mAP50:             {metrics.box.map50:.4f}")
