from src.data_loader import load_image, preprocess_image, reconstruct_image
# from src.canopy_classification.scikit_img import crown_pixel_mask_from_ndvi_tiles, label_crowns_in_ndvi_tiles
from src.pipeline import estimate_chm_from_dsm
# from src.visualisation.visualise_tiles import show_labeled_tile
import matplotlib.pyplot as plt
import numpy as np


# for detecttron2
if __name__ == "__main__":
    path = 'data/raw/20250728-ISA-NDVI-10cm.tif'
    dsm_img, mask = load_image(path)
    print(f"Loaded image shape: {dsm_img.shape}")

    dsm_tiles = preprocess_image(dsm_img, tile_size=512, overlap=64)

    # chm_tiles = estimate_chm_from_dsm(dsm_tiles)
    # print(f"Estimated CHM tiles shape: {chm_tiles.shape}")
    # assert dsm_tiles.shape == chm_tiles.shape, "Tile shapes must match exactly!"

    # visualise some tiles
    # for i in range(2, min(4, chm_tiles.shape[0])):
    #     for j in range(2, min(4, chm_tiles.shape[1])):
    #         print(f"Tile ({i}, {j})")
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(chm_tiles[i, j])
    #         plt.title(f"CHM Tile ({i}, {j})")
    #         plt.axis("off")
    #         plt.show()

    # reconstruct a samples of CHM tiles
    # small_recon = reconstruct_image(chm_tiles[3:10, 3:10], dsm_img.shape, pad_dims, tile_size=512, overlap=64)
    # print(f"Reconstructed CHM shape: {small_recon.shape}")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(small_recon)
    # plt.title("Reconstructed CHM (from 4 tiles)")
    # plt.axis("off")
    # plt.show()



    # Create a simple mask (replace this with your actual mask)
    

    dsm_image = reconstruct_image(dsm_tiles, dsm_img.shape, mask, tile_size=512, overlap=64)
    print(np.nanmax(np.abs(dsm_image - dsm_img)))
    # visualise dsm_image
    plt.figure(figsize=(10, 10))
    plt.imshow(dsm_image)
    plt.title("Estimated DSM")
    plt.axis("off")
    plt.show()





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