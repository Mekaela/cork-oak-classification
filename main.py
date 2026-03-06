from src.data_loader import load_dsm_with_mask, preprocess_image, reconstruct_image
# from src.canopy_classification.scikit_img import crown_pixel_mask_from_ndvi_tiles, label_crowns_in_ndvi_tiles
from src.pipeline import compute_chm_from_dsm, save_chm
# from src.visualisation.visualise_tiles import show_labeled_tile
import matplotlib.pyplot as plt
import numpy as np



# for detecttron2
if __name__ == "__main__":
    dsm_path = 'data/raw/20250728-ISA-dsm-10cm.tif'
    # ndvi_path = 'data/raw/20250728-ISA-NDVI-10cm.tif'
    dsm, mask, profile = load_dsm_with_mask(dsm_path)

    # only needed when creating CHM. Use CHM image from now on from data
    # chm = compute_chm_from_dsm(dsm, mask, window_size=100, nodata=0.0)
    

    # visualise some tiles
    # for i in range(0, min(2, chm_tiles.shape[0])):
    #     for j in range(0, min(2, chm_tiles.shape[1])):
    #         print(f"Tile ({i}, {j})")
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(chm_tiles[i, j])
    #         plt.title(f"CHM Tile ({i}, {j})")
    #         plt.axis("off")
    #         plt.show()

    # reconstruct a samples of CHM tiles
    # small_recon = reconstruct_image(chm_tiles[0:10, 0:10], dsm_img.shape, mask, tile_size=512, overlap=64)
    # print(f"Reconstructed CHM shape: {small_recon.shape}")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(small_recon)
    # plt.title("Reconstructed CHM (from 4 tiles)")
    # plt.axis("off")
    # plt.show()

    # chm_full = estimate_chm_from_dsm_tile(dsm_img) 
    # tiles = preprocess_image(dsm_img, tile_size=512, overlap=64)
    # chm_recon = reconstruct_image(chm_tiles, dsm_img.shape, mask, tile_size=512, overlap=64)
    # diff = np.nanmax(np.abs(chm_recon - chm_full))
    # print(f"Max absolute difference between reconstructed CHM and full CHM: {diff}")





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