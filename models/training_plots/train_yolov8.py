from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os


MODEL_PATH = "runs/detect/yolov8_plots_run-10/weights/best.pt"
INPUT_IMAGE = "data/plots/plot10/plot10_RGB_3band.tif"
# INPUT_IMAGE = "data/raw/20250728-ISA-RGB-10cm_3band.tif"
OUTPUT_DIR = "runs/detect"                # Where to save results
TILE_SIZE = 640                                 # Must match your training size (usually 640)
OVERLAP = 100                                   # Overlap between tiles to prevent cutting objects
CONF_THRESHOLD = 0.03

def train_yolov8():
    # Load the YOLOv8 model (you can choose 'yolov8n.pt', 'yolov8s.pt', etc.)
    model = YOLO("yolov8n.pt")

    # Train using the dataset config in the same folder
    data_cfg = Path(__file__).parent / "data.yaml"

    results = model.train(data=str(data_cfg), epochs=50, imgsz=640, batch=8, name="yolov8_plots_run")


def predict_yolov8():
    # 1. Load your trained model
    # Replace 'runs/detect/exp/weights/best.pt' with your actual path
    
    model = YOLO(MODEL_PATH)

    # 2. Set the source image
    source_image = INPUT_IMAGE

    # 3. Run prediction
    # save=True saves the result image to runs/detect/predict/
    # save_txt=True saves the coordinates to runs/detect/predict/
    results = model.predict(
        source=source_image, 
        save=True, 
        save_txt=True, 
        conf=CONF_THRESHOLD,  # Confidence threshold (0.25 is standard, lower = more detections)
        verbose=True
    )

    # 4. (Optional) Access results programmatically
    for result in results:
        print(f"Image: {result.path}")
        print(f"Boxes detected: {len(result.boxes)}")
        
        # Get coordinates for each box
        for box in result.boxes:
            # xyxy: [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist() 
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            print(f"  Tree found: Class {cls}, Confidence: {conf:.2f}, Box: {xyxy}")

# this is shit, it gets stuck in a loop.
def predict_large_image(model_path, input_img, output_dir, tile_size=640, overlap=100, conf=0.15):
    # Load model
    model = YOLO(model_path)
    
    # Read image
    img = cv2.imread(input_img)
    if img is None:
        print(f"Error: Could not read {input_img}")
        return

    h, w, _ = img.shape
    print(f"Processing image: {h}x{w}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate steps
    step_x = tile_size - overlap
    step_y = tile_size - overlap

    # List to store all detections
    all_detections = []

    # Sliding Window
    y_start = 0
    while y_start < h:
        x_start = 0
        while x_start < w:
            # Define crop boundaries
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            # Extract tile
            tile = img[y_start:y_end, x_start:x_end]
            
            # Run prediction on tile
            results = model.predict(
                source=tile,
                conf=conf,
                verbose=False,
                imgsz=tile_size  # Explicitly set size
            )
            
            # Extract boxes and adjust coordinates
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Adjust coordinates to global image space
                    global_x1 = x_start + x1
                    global_y1 = y_start + y1
                    global_x2 = x_start + x2
                    global_y2 = y_start + y2
                    
                    all_detections.append({
                        "x1": global_x1, "y1": global_y1,
                        "x2": global_x2, "y2": global_y2,
                        "conf": conf, "cls": cls
                    })
            
            x_start += step_x
            if x_start + tile_size > w and x_start < w:
                x_start = w - tile_size # Adjust for last tile
        
        y_start += step_y
        if y_start + tile_size > h and y_start < h:
            y_start = h - tile_size

    # Draw all boxes on the original image
    result_img = img.copy()
    for det in all_detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = f"Cork Oak {det['conf']:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result
    output_path = os.path.join(output_dir, "result_large.tif")
    cv2.imwrite(output_path, result_img)
    
    print(f"✅ Done! Found {len(all_detections)} trees.")
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    # predict_large_image(MODEL_PATH, INPUT_IMAGE, OUTPUT_DIR, tile_size=TILE_SIZE, overlap=OVERLAP, conf=CONF_THRESHOLD)
    predict_yolov8()