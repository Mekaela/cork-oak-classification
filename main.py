from src.data_loader import load_tif_with_mask, preprocess_image, reconstruct_image
from src.canopy_classification.scikit_img import crown_pixel_mask_from_ndvi_tiles, label_crowns_in_ndvi_tiles
from src.pipeline import build_chm_mask, compute_chm_from_dsm, save_layer, estimate_dtm_from_dsm
from src.visualisation.visualise_tiles import show_labeled_tile
import matplotlib.pyplot as plt
import numpy as np


# for detecttron2
if __name__ == "__main__":
    dsm_path = 'data/raw/20250728-ISA-dsm-10cm.tif'
    # ndvi_path = 'data/raw/20250728-ISA-NDVI-10cm.tif'
    dsm, mask, profile = load_tif_with_mask(dsm_path)
    # dsm tested and works fine
    # save_layer(dsm, profile, "data/processed/20250728-ISA-dsm-10cm-01.tif", nodata=0.0)
    # window_size = 201
    # dtm = estimate_dtm_from_dsm(dsm, mask, window_size=window_size)
    # save_layer(dtm, profile, "data/processed/20250728-ISA-dtm-10cm-06.tif", nodata=0.0)


    # only needed when creating CHM. Use CHM image from now on from data
    # chm = compute_chm_from_dsm(dsm, dtm, mask, nodata=0.0)
    # save_layer(chm, profile, "data/processed/20250728-ISA-chm-10cm-04.tif", nodata=0.0)

    chm_path = 'data/processed/20250728-ISA-chm-10cm.tif'
    chm = load_tif_with_mask(chm_path)[0]
    chm_mask = build_chm_mask(chm)
    save_layer(chm_mask[:, :, np.newaxis].astype(np.uint8), profile, "data/processed/20250728-ISA-chm-mask-10cm-2m.tif", nodata=0.0)



# For scikit_image
# if __name__ == "__main__":
#     path = 'data/raw/20250728-ISA-NDVI-10cm.tif'
#     corkImg = load_image(path)
#     print(f"Loaded image shape: {corkImg.shape}")

#     ndvi_tiles = preprocess_image(corkImg, tile_size=512, overlap=64)
#     print(f"Preprocessed tiles shape: {ndvi_tiles.shape}")

#     crown_mask_tiles = crown_pixel_mask_from_ndvi_tiles(ndvi_tiles, threshold=0.55)
#     print(f"Crown mask tiles shape: {crown_mask_tiles.shape}")

#     labeled_crown_tiles = label_crowns_in_ndvi_tiles(crown_mask_tiles, min_pixels=20)
#     print(f"Labeled crown tiles shape: {labeled_crown_tiles.shape}")

#     for i in range(0, min(3, labeled_crown_tiles.shape[0])):
#         for j in range(0, min(3, labeled_crown_tiles.shape[1])):
#             print(f"Tile ({i}, {j})")
#             show_labeled_tile(ndvi_tiles[i, j], labeled_crown_tiles[i, j])