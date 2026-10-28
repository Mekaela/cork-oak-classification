from ultralytics import YOLO
from pathlib import Path


def train_yolo():
    # Load the YOLOv8 model (you can choose 'yolov8n.pt', 'yolov8s.pt', etc.)
    model = YOLO("yolov8n-seg.pt")

    # Train using the dataset config in the same folder
    data_cfg = Path(__file__).parent / "data.yaml"

    results = model.train(
        data=str(data_cfg), 
        epochs=10, 
        # patience=50, 
        imgsz=640, 
        batch=8, 
        name="yolo_plots_run", 
        overlap_mask=True,
        save_json=True,
    )


if __name__ == "__main__":
    train_yolo()
