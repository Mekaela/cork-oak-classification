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



# tiles have to be similar to the training data, which was 100m, with 30m buffer
site_path = "data/lezirias/"
img_path = site_path + "/raw/20250728-ISA-RGB-10cm.tif"
tiles_path = site_path + "/tiles/"
# predictions_path = site_path + "/montado2/tiles/"
# predictions_path = site_path + "/urban/tiles/"
predictions_path = site_path + "tiles/test/"

def tile_rgb():
    # Set tiling parameters
    buffer = 30
    tile_width = 40
    tile_height = 40
    threshold = 0.1 
    
    return tile_data(img_path, tiles_path, buffer, tile_width, tile_height, threshold, mode="rgb")

# takes a while to run. Saves to base folder ./predictions
def predict(tiles_path):
    # trained_model = 'models/predicting/250312_flexi.pth'
    # trained_model = 'models/weights/model_5_detectree2_2.pth'
    trained_model = 'data/train_outputs/model_final.pth'
    cfg = setup_cfg(update_model=trained_model)
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg) 
    predict_on_data(tiles_path, predictor=predictor)

def project(tiles_path, predictions_path):
    # project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    project_to_geojson(tiles_path,  "predictions/", predictions_path + "predictions_geo/")

 
def stitch():
    # crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
    crowns = stitch_crowns(predictions_path + "predictions_geo/", 1)
    clean = clean_crowns(crowns, 0.8, confidence=0.1)
    # clean = clean[clean["Confidence_score"] > 0.2]
    clean.to_file(site_path + "/crowns_out_site_02.gpkg")

# tile_rgb()
predict(tiles_path)
project(tiles_path, predictions_path)
stitch()

