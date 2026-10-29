from src.data_preparation.data_loader import load_tif_with_mask
from src.data_preparation.layer_prep import compute_chm_from_dsm, estimate_dtm_from_dsm, save_layer
from src.models.predicting.predict_yolo import predict_large_image

MODEL = "YOLO"
RGB = "data/raw/20250728-ISA-RGB-10cm.tif"
DSM = "data/raw/20250728-ISA-dsm-10cm.tif"
OUTPUT_DIR = "src/models/predicting"

# prepare image, detect crowns, predict heights and crown area
# no training is done here, pipeline is for predicting
if __name__ == "__main__":
    # load the RGB image and the DSM (Digital Surface Model) for height information
    # rgb_image, rgb_mask = load_tif_with_mask(RGB)
    dsm, dsm_mask, profile = load_tif_with_mask(DSM)
    # create the chm
    window_size = 201
    dtm = estimate_dtm_from_dsm(dsm, dsm_mask, window_size=window_size)
    # save_layer(dtm, profile, "data/processed/20250728-ISA-dtm-10cm-06.tif", nodata=0.0)

    chm = compute_chm_from_dsm(dsm, dtm, dsm_mask, nodata=0.0)
    # save_layer(chm, profile, "data/processed/20250728-ISA-chm-10cm-11.tif", nodata=0.0)
    
    if MODEL == "YOLO":
        model_path = "src/models/weights/best_yolov8.pt"
        # run the YOLO prediction (for large images, so needs tiling)
        gdf = predict_large_image(model_path, RGB, OUTPUT_DIR, slice_size=640, overlap_ratio=0.1, conf=0.298, device="cpu")
        gdf
        print('YOLO finished')
    elif MODEL == "Detectree2":
        print()

    # for each predicted polygon, estimate the height and crown areas