from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import unary_union
from shapely.validation import make_valid

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon

# this is from the site run, change to predict plots
MODEL_PATH = "src/models/weights/best_yolov8.pt"
model = 'yolo_site'

OUTPUT_DIR = f"src/validation/{model}"                     # Where results are
TILE_SIZE = 640                                 # Must match training size (usually 640)
OVERLAP = 100                                   # Overlap between tiles to prevent cutting objects
# each trained model has a different optimum threshold on the validation data. Find this in the run's MaskF1_curve
CONF_THRESHOLD = 0.298

def predict_yolo(input_image, output_path):
    # Load  trained model
    model = YOLO(MODEL_PATH)

    # Set the source image
    source_image = input_image
    # need to fix this to save properly
    output_gpkg = Path(OUTPUT_DIR) / f"{output_path}.gpkg"
    # output_gpkg = Path(OUTPUT_DIR) / "yolo_plot61_1004_040.gpkg"

    # Run prediction
    # save=True saves the result image to runs/detect/predict/
    # save_txt=True saves the coordinates to runs/detect/predict/
    results = model.predict(
        source=source_image, 
        save=True, 
        save_txt=True, 
        conf=CONF_THRESHOLD,
        verbose=True
    )

    all_features = []

    # Access results programmatically
    for result in results:
        print(f"Image: {result.path}")
        print(f"Boxes detected: {len(result.boxes)}")

        # Get georeferencing from the source raster.
        with rasterio.open(source_image) as src:
            transform = src.transform
            crs = src.crs

        if result.masks is None:
            print("No segmentation masks were returned.")
            continue

        polygons = result.masks.xy
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, pixel_points in enumerate(polygons):
            if len(pixel_points) < 3:
                continue

            # Convert YOLO pixel coordinates (x, y) to map coordinates.
            map_points = [
                transform * (float(px), float(py))
                for px, py in pixel_points
            ]

            polygon = Polygon(map_points)

            if polygon.is_empty or not polygon.is_valid:
                polygon = polygon.buffer(0)

                if polygon.is_empty:
                    continue

            all_features.append({
                "class_id": int(classes[i]),
                "confidence": float(confidences[i]),
                "geometry": polygon,
            })

    if not all_features:
        print("No valid polygons to save.")
        return None

    gdf = gpd.GeoDataFrame(all_features, crs=crs)
    gdf["tree_id"] = range(len(gdf))

    # map class IDs to readable names.
    class_names = {
        0: "tree_crown",
    }
    gdf["class_name"] = gdf["class_id"].map(class_names).fillna("unknown")

    gdf = gdf[[
        "tree_id",
        "class_id",
        "class_name",
        "confidence",
        "geometry",
    ]]

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(
        output_gpkg,
        layer="tree_crowns",
        driver="GPKG",
    )

    print(f"Saved {len(gdf)} polygons to: {output_gpkg}")
    print(f"CRS: {crs}")

    return gdf

def predict_large_image(
    model_path,
    input_img,
    output_dir,
    slice_size=640,
    overlap_ratio=0.2,
    conf=0.25,
    device="cpu",
):
    """
    Run YOLO segmentation on a large georeferenced image using SAHI,
    then convert detected masks to a map-coordinate GeoPackage.
    """
    os.makedirs(output_dir, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    with rasterio.open(input_img) as src:
        transform = src.transform
        crs = src.crs

        img_array = src.read()

        if src.count == 1:
            img_array = np.stack([img_array[0]] * 3, axis=0)
        elif src.count >= 3:
            img_array = img_array[:3]

        # CHW -> HWC
        img_array = np.transpose(img_array, (1, 2, 0))

        if img_array.dtype != np.uint8:
            band_min = img_array.min(axis=(0, 1), keepdims=True)
            band_max = img_array.max(axis=(0, 1), keepdims=True)
            img_array = (
                255
                * (img_array - band_min)
                / np.maximum(band_max - band_min, 1)
            ).astype(np.uint8)

        # SAHI expects RGB.
        image = img_array

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.2,
    )

    records = []

    for prediction in result.object_prediction_list:
        mask = prediction.mask

        if mask is None:
            continue

        # SAHI stores one or more COCO polygons:
        # [[x1, y1, x2, y2, ...], ...]
        for segmentation in mask.segmentation:
            coords = np.asarray(segmentation, dtype=float).reshape(-1, 2)

            if len(coords) < 3:
                continue

            geo_coords = [
                transform * (float(x), float(y))
                for x, y in coords
            ]

            polygon = Polygon(geo_coords)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if polygon.is_empty:
                continue

            records.append(
                {
                    "geometry": polygon,
                    "confidence": float(prediction.score.value),
                    "class_id": int(prediction.category.id),
                    "class_name": prediction.category.name,
                    "area_px": polygon.area,
                }
            )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf["tree_id"] = range(len(gdf))

    output_gpkg = Path(output_dir) / "tree_crowns_predicted.gpkg"
    gdf.to_file(output_gpkg, layer="tree_crowns", driver="GPKG")

    print(f"Saved {len(gdf)} crowns to {output_gpkg}")

    return gdf


if __name__ == "__main__":
    # predict_large_image(MODEL_PATH, "data/raw/20250728-ISA-RGB-10cm.tif", OUTPUT_DIR, slice_size=TILE_SIZE, overlap_ratio=0.2, conf=CONF_THRESHOLD)
    
    plot_names = ["plot61","plot175","plot238"] 
    for plot in plot_names:
        input_image = f"data/plots/{plot}/{plot}_RGB_3band.tif"
        predict_yolo(input_image, plot)