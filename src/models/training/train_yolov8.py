from black import output
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon


MODEL_PATH = "runs_yolo/runs_yolo8_site_300epochs/segment/yolo_plots_run/weights/best.pt"
model = YOLO(MODEL_PATH)

results = model.val(
    data="models/training_plots/data.yaml",
    split="val",          # or "val"
    save_json=True,
    save_conf=True,
    conf=0.001,            # keep low-confidence detections for PR curve
    iou=0.7,               # NMS IoU, not the mAP IoU threshold
    imgsz=640,             # use the same size used in training/evaluation
)


def train_yolo():
    # Load the YOLOv8 model (you can choose 'yolov8n.pt', 'yolov8s.pt', etc.)
    model = YOLO("yolov8n-seg.pt")

    # Train using the dataset config in the same folder
    data_cfg = Path(__file__).parent / "data.yaml"

    results = model.train(
        data=str(data_cfg), 
        epochs=135, 
        # patience=50, 
        imgsz=640, 
        batch=8, 
        name="yolo_plots_run", 
        overlap_mask=True,
        save_json=True,
    )


# if __name__ == "__main__":
    # train_yolo()
    
    # plot_names = ["plot61","plot175","plot238"] 
    # for plot in plot_names:
    #     input_image = f"data/plots/{plot}/{plot}_RGB_3band.tif"
    #     predict_yolo(input_image, {plot})
