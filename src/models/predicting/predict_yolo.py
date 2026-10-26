from black import output
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

# this is from the training run, alternatively can save path in weights folder with other weights
MODEL_PATH = "runs_yolo/runs_yolo8_site_300epochs/segment/yolo_plots_run/weights/best.pt"
model = 'yolo_plots'

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

# note to self: check if this works:
def predict_large_image(model_path, input_img, output_dir, tile_size=TILE_SIZE, overlap=OVERLAP, conf=CONF_THRESHOLD):
    import rasterio
    import geopandas as gpd
    from shapely.geometry import Polygon
    import pandas as pd
    
    model = YOLO(model_path)
    
    # Read image WITH georeferencing first
    with rasterio.open(input_img) as src:
        transform = src.transform
        crs = src.crs
        img_array = src.read()
        
        # Handle single or multi-band
        if src.count == 1:
            img_array = np.stack([img_array[0]] * 3, axis=-1)
        else:
            img_array = np.transpose(img_array, (1, 2, 0))
        
        # Convert to uint8 if needed
        if img_array.dtype != np.uint8:
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Rasterio reads as RGB, OpenCV expects BGR
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if img is None:
        print(f"Error: Could not read {input_img}")
        return

    h, w, _ = img.shape
    print(f"Processing image: {h}x{w}")

    os.makedirs(output_dir, exist_ok=True)

    step_x = tile_size - overlap
    step_y = tile_size - overlap

    all_detections = []

    # Sliding window
    y_start = 0
    while y_start < h:
        x_start = 0
        while x_start < w:
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            tile = img[y_start:y_end, x_start:x_end]
            
            if tile.size == 0:
                x_start += step_x
                continue
            
            results = model.predict(
                source=tile,
                conf=conf,
                verbose=False,
                imgsz=tile_size
            )
            
            for result in results:
                if result.masks is not None:
                    polygons = result.masks.xy
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for i, poly in enumerate(polygons):
                        if confidences[i] < conf:
                            continue
                        
                        # Scale polygon to actual tile dimensions
                        tile_h, tile_w = tile.shape[:2]
                        scale_x = tile_w / tile_size
                        scale_y = tile_h / tile_size
                        
                        scaled_poly = poly.copy()
                        scaled_poly[:, 0] *= scale_x
                        scaled_poly[:, 1] *= scale_y
                        
                        # Shift to global pixel coordinates
                        global_points = scaled_poly + np.array([x_start, y_start])
                        
                        all_detections.append({
                            "points": global_points,
                            "conf": float(confidences[i]),
                            "cls": int(classes[i])
                        })
            
            x_start += step_x
            if x_start >= w:
                break
        
        y_start += step_y
        if y_start >= h:
            break

    # Convert pixel coordinates to geographic coordinates and create polygons
    geographic_polygons = []
    
    for det in all_detections:
        points = det["points"]
        
        # Transform pixel coordinates to geographic coordinates
        # rasterio.transform.xy converts row, col to x, y
        # But our points are in (x, y) pixel space, so we need to use the transform directly
        geo_coords = []
        for px, py in points:
            # Transform from pixel (col, row) to geographic (x, y)
            gx, gy = transform * (px, py)
            geo_coords.append((gx, gy))
        
        # Create Shapely Polygon
        if len(geo_coords) >= 3:  # Need at least 3 points for a polygon
            polygon = Polygon(geo_coords)
            
            geographic_polygons.append({
                "geometry": polygon,
                "confidence": det["conf"],
                "class": det["cls"],
                "area_px": polygon.area  # Optional: area in pixel units
            })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geographic_polygons, crs=crs)
    
    # Optional: Add additional attributes
    gdf["tree_id"] = range(len(gdf))
    gdf = gdf[["tree_id", "class", "confidence", "area_px", "geometry"]]
    
    print(f"Created {len(gdf)} polygon features")

    # Save to GeoPackage
    output_gpkg = os.path.join(output_dir, "tree_polygons_20261002_080_site.gpkg")
    gdf.to_file(output_gpkg, driver="GPKG", layer="tree_crowns")
    
    print(f"✅ Done! Found {len(gdf)} trees (polygons).")
    print(f"GeoPackage saved to: {output_gpkg}")
    print(f"CRS: {crs}")
    
    # Optional: Also save as GeoJSON for easy viewing
    output_geojson = os.path.join(output_dir, "tree_polygons_20261002_080_site.geojson")
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"GeoJSON saved to: {output_geojson}")
    
    return gdf


if __name__ == "__main__":
    # predict_large_image(MODEL_PATH, INPUT_IMAGE, OUTPUT_DIR, tile_size=TILE_SIZE, overlap=OVERLAP, conf=CONF_THRESHOLD)
    
    plot_names = ["plot61","plot175","plot238"] 
    for plot in plot_names:
        input_image = f"data/plots/{plot}/{plot}_RGB_3band.tif"
        predict_yolo(input_image, plot)