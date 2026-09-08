from detectree2.preprocessing.tiling import tile_data
import geopandas as gpd
import rasterio
from pathlib import Path
from detectree2.models.train import setup_cfg
from detectree2.models.train import register_train_data
from detectree2.preprocessing.tiling import to_traintest_folders
from detectree2.models.train import register_train_data
from detectree2.models.train import MyTrainer
from detectree2.models.train import load_json_arr
from detectree2.models.predict import predict_on_data
from detectron2.engine import DefaultPredictor
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
import json
import matplotlib.pyplot as plt



# input paths
SITE_PATH = "data/lezirias"
IMG_PATH = SITE_PATH + "/20250728-ISA-RGB-10cm_3band.tif"
CROWN_PATH = SITE_PATH + "/Crown_training_polygons_60ha.gpkg"
TRAIN_LOCATION = SITE_PATH + "/tiles/train/"
TEST_LOCATION = SITE_PATH + "/tiles/test/"
trained_model = 'data/train_outputs/model_final.pth'

def update_pretrained_model():
    trained_model = "models/predicting/250312_flexi.pth"

    trains = ("lezirias_train",)
    tests = ("lezirias_val",)

    out_dir = "data/train_outputs"

    cfg = setup_cfg(update_model=trained_model, trains=trains, tests=tests, workers=4, eval_period=100, max_iter=3000, out_dir=out_dir)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.DEVICE = "cpu"

    trainer = MyTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=False)
    trainer.train()

def post_training_analysis():
    experiment_folder = "data/train_outputs"
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], label='Total Validation Loss', color='red')
    plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], label='Total Training Loss')

    plt.legend(loc='upper right')
    plt.title('Comparison of the training and validation loss of detectree2')
    plt.ylabel('Total Loss')
    plt.xlabel('Number of Iterations')
    plt.show()

    plt.plot(
   [x['iteration'] for x in experiment_metrics if 'bbox/AP50' in x],
   [x['bbox/AP50'] for x in experiment_metrics if 'bbox/AP50' in x], label='Validation AP50')

    plt.legend(loc='lower right')
    plt.title('Validation AP50 over training iterations')
    plt.ylabel('AP50')
    plt.xlabel('Number of Iterations')
    plt.show()

# takes a while to run. Saves to base folder ./predictions
def predict(trained_model):
    cfg = setup_cfg(update_model=trained_model)
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    predict_on_data(TEST_LOCATION, predictor=predictor)

if __name__ == "__main__":
    data = rasterio.open(IMG_PATH)
    crowns = gpd.read_file(CROWN_PATH)

    # Set tiling parameters
    buffer = 30
    tile_width = 40
    tile_height = 40
    threshold = 0.05
    data_folder = SITE_PATH + "/tiles/"

    # tile_data(IMG_PATH, data_folder, buffer, tile_width, tile_height, crowns, threshold, mode="rgb", tile_placement="grid", nan_threshold=0.9)
    # to_traintest_folders(data_folder, data_folder, test_frac=0.15, strict=False, folds=5)
    # register_train_data(TRAIN_LOCATION, 'lezirias', val_fold=5)
    # update_pretrained_model() 
    # post_training_analysis()
    # note to self: check detectree2/notebooks/colab/evaluationJB.ipynb
    # for more evaluation
    # predict(trained_model)
    # project_to_geojson(TEST_LOCATION,  "predictions/", TEST_LOCATION + "predictions_geo/")
    crowns = stitch_crowns(predictions_path + "predictions_geo/", 1)
    clean = clean_crowns(crowns, 0.8, confidence=0.1)
    # clean = clean[clean["Confidence_score"] > 0.2]
    clean.to_file(site_path + "/crowns_out_urban_08_01.gpkg")



# this is only for visualising tiles. not pertanent to training
# it shows bounding boxes of crowns on the tiles, so could be useful for drawing bounding boxes.
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectree2.models.train import combine_dicts, register_train_data
import random
import cv2
from PIL import Image
from IPython.display import display

def display_training_tiles(site, fold=1):
    name = site
    train_location = "data/" + name + "/tiles" + "/train"
    dataset_dicts = combine_dicts(train_location, fold)
    trees_metadata = MetadataCatalog.get(name + "_train")

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=trees_metadata, scale=0.3)
        out = visualizer.draw_dataset_dict(d)
        image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        cv2.imshow("Output", image)
        # the value is the time in milliseconds to keep the window open. 0 means wait indefinitely until a key is pressed.
        cv2.waitKey(0)
        cv2.destroyAllWindows()
