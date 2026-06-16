from detectree2.preprocessing.tiling import tile_data
import geopandas as gpd
import rasterio
from pathlib import Path

# to create tiles from the training data
# training_sites = [
#     {
#         "img_path": "data/plots/plot10/plot10_RGB_3band.tif",
#         "crown_path": "data/plots/plot10/plot10_crowns.gpkg",
#     },
# ]


# replace as 3 band RGB
# src_path = "data/plots/plot10/plot10_RGB.tif"
# dst_path = "data/plots/plot10/plot10_RGB_3band.tif"
train_sites = ["plot10", "plot14", "plot39", "plot43", "plot88", "plot143", "plot183", "plot205", "plot242"]
val_sites = ["plot61", "plot175", "plot238"]

def four_to_three_band(src_path, dst_path):
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=3)

        rgb = src.read([1, 2, 3])

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(rgb)

# for site in train_sites:
#     src_path = f"data/plots/{site}/{site}_RGB.tif"
#     dst_path = f"data/plots/{site}/{site}_RGB_3band.tif"
#     four_to_three_band(src_path, dst_path)



def create_training_tiles(sites, output_dir="data/plots/tiles/train/"):
    for site in sites:
        img_path = f"data/plots/{site}/{site}_RGB_3band.tif"
        crown_path = f"data/plots/{site}/{site}_crowns.gpkg"
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

# create_training_tiles(val_sites, output_dir="data/plots/tiles/")

# from detectree2.preprocessing.tiling import to_traintest_folders

# data_folder = "data/plots/tiles/"
# to_traintest_folders(data_folder, data_folder, test_frac=0.25, strict=False)



from detectree2.models.train import register_train_data

# train_location = "data/plots/tiles"
train_location = "data/plots/tiles/"
register_train_data(train_location, "Lizirias", val_fold=1)


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



# from pathlib import Path
# import cv2

# p = Path("data/plots/tiles/train/plot39_RGB_3band_-59244_-93923_40_10_3763.png")
# print(p.exists(), p.is_file())
# img = cv2.imread(str(p))
# print(img is None)




# To apply pre-determined weights with the training data
from detectree2.models.train import setup_cfg

# Set the base (pre-trained) model from the detectron2 model_zoo
base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

trains = ("Lizirias_train",)
tests = ("Lizirias_val",)

out_dir = "./train_outputs"

cfg = setup_cfg(base_model, trains, tests, workers=4, eval_period=100, max_iter=3000, out_dir=out_dir)
cfg.MODEL.DEVICE = "cpu"
cfg.DATALOADER.NUM_WORKERS = 0

# # training the model
from detectree2.models.train import MyTrainer

trainer = MyTrainer(cfg, patience=5)
trainer.resume_or_load(resume=False)

from detectron2.data import DatasetCatalog
print(DatasetCatalog.list())

trainer.train()