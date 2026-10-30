import numpy as np
import detectree2
import rasterio
import geopandas as gpd
from shapely.geometry import box
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime


# input paths
SITE_PATH = "data"
TILES_PATH = SITE_PATH + "/tiles"
TEST_LOCATION = TILES_PATH + "/test"
IMG_PATH = SITE_PATH + "/raw/20250728-ISA-RGB-10cm_3band.tif"
PRED_PATH = SITE_PATH + "montado_trained_site/"
# PRED_PATH = SITE_PATH + "/urban/tiles/"
OUT_FILE = SITE_PATH + f"/crowns_out_montado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpkg"

# tiles have to be similar to the training data, which was 100m, with 30m buffer
def tile_rgb(img_path=IMG_PATH, tiles_path=TILES_PATH):
    # Set tiling parameters
    buffer = 10
    tile_width = 40
    tile_height = 40
    threshold = 0.1 
    return tile_data(img_path, tiles_path, buffer, tile_width, tile_height, threshold=threshold, mode="rgb", nan_threshold=0.9, tile_placement="adaptive")

def predict(trained_model, tiles_path, test_location=TEST_LOCATION):
    cfg = setup_cfg(update_model=trained_model)
    cfg.MODEL.DEVICE = "cpu" 
    predictor = DefaultPredictor(cfg)
    predict_on_data(test_location, tiles_path, predictor=predictor)

def project(tiles_path, predictions_path):
    # project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    project_to_geojson(tiles_path,  "predictions/", predictions_path + "predictions_geo/")
 
def stitch(out_file=OUT_FILE, pred_path=PRED_PATH):
    # crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
    crowns = stitch_crowns(pred_path + "predictions_geo/", 1)
    clean = clean_crowns(crowns, 0.8, confidence=0.50)
    # clean = clean[clean["Confidence_score"] > 0.2]
    clean.to_file(out_file)
    return clean

tile_rgb(IMG_PATH, TILES_PATH)
# predict('src/models/weights/model_17_dt2_site.pth', TEST_LOCATION)
# project(TILES_PATH, PRED_PATH)
# stitch(OUT_FILE)

