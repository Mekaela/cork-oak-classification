from detectree2.preprocessing.tiling import tile_data
import geopandas as gpd
import rasterio
from detectree2.models.train import setup_cfg
from detectree2.models.train import register_train_data
from detectree2.preprocessing.tiling import to_traintest_folders
from detectree2.models.train import MyTrainer
import matplotlib.pyplot as plt
from detectree2.models.train import load_json_arr
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectree2.models.train import combine_dicts, register_train_data
import cv2
from IPython.display import display

from src.models.predicting.predict_dt2 import IMG_PATH


def create_training_tiles(output_dir, img_path, crown_path):
        # Read crowns and ensure CRS matches the raster
        with rasterio.open(img_path) as raster:
            crowns = gpd.read_file(crown_path).to_crs(raster.crs)
            print(f"Processing {img_path} with {len(crowns)} crowns")

            tile_data(
                img_path=img_path,
                out_dir=output_dir,
                crowns=crowns,
                mode="rgb",
                tile_placement="adaptive",
                # additional_nodata=[-10000, -20000],
                tile_width=40,
                buffer=10,
                # other parameters...
                tile_height=40,
                threshold=0.0,
                nan_threshold=0.9,
            )


def create_tiles_across_sites(sites, data_root="data/plots"):
    for site in sites:
        img_path = f"data/plots/{site}/{site}_RGB_3band.tif"
        crown_path = f"data/plots/{site}/{site}_crowns.gpkg"
        create_training_tiles(output_dir=f"{data_root}/tiled/", img_path=img_path, crown_path=crown_path)


# check if it's looking for the correct train and val directories
# from detectron2.data import DatasetCatalog

# ds = DatasetCatalog.get("Lizirias_train")
# print('train')
# print(len(ds))
# print(ds[:1])
# dval = DatasetCatalog.get("Lizirias_val")
# print('val')
# print(len(dval))
# print(dval[:1])


# check if image files can be read
# from pathlib import Path
# import cv2

# p = Path("data/plots/tiles/train/plot39_RGB_3band_-59244_-93923_40_10_3763.png")
# print(p.exists(), p.is_file())
# img = cv2.imread(str(p))
# print(img is None)


def update_pretrained_model():
    trained_model = "src/models/weights/250312_flexi.pth"

    trains = ("lezirias_train",)
    tests = ("lezirias_val",)

    out_dir = "data/train_outputs"

    cfg = setup_cfg(update_model=trained_model, trains=trains, tests=tests, workers=0, eval_period=100, max_iter=3000, out_dir=out_dir)
    # needed if not using cuda. Otherwise, it will try to use GPU and fail if not available.
    cfg.DATALOADER.NUM_WORKERS = 0
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


# sample usage for plots
if __name__ == "__main__":
    train_sites = ["plot10", "plot14", "plot39", "plot43", "plot88", "plot143", "plot183", "plot205", "plot242"]
    val_sites = ["plot61", "plot175", "plot238"]

    img_path="data/raw/training_clipped_RGB.tif"
    crown_path="data/raw/clipped_training_polygons_60ha.gpkg"

    # change to val_sites or train_sites depending on which you want to create tiles for
    # for plots:
    # create_tiles_across_sites(train_sites, data_root="data/plots")
    # for one image - entire site:
    create_training_tiles(output_dir="data/tiles", img_path=img_path, crown_path=crown_path)

    to_traintest_folders(
            tiles_folder="data/plots/tiled/",
            out_folder="data/plots/tiled/",
            test_frac=0.0,
            strict=False,
            folds=5,
            )

    register_train_data(
            str("data/plots/tiled/train"),
            "lezirias",
            val_fold=5,
        )
    update_pretrained_model()

