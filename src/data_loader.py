# Load and preprocess images (rasterio, opencv)
import os
import numpy as np
import rasterio
from skimage.util import view_as_windows
# import cv2  
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def load_image(image_path):
    try:
        with rasterio.open(image_path) as src:
            image = src.read()  # Read all bands
            image = np.transpose(image, (1, 2, 0))  # HWC
        return image
    except Exception as e:
        raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")

def preprocess_image(image, tile_size=512, overlap=64):
    step = tile_size - overlap
    # image: (H, W, C)
    tiles = view_as_windows(image, (tile_size, tile_size, image.shape[2]), step=step)
    # tiles shape: (n_rows, n_cols, 1, tile_size, tile_size, C)
    tiles = tiles.squeeze(axis=2)  # (n_rows, n_cols, tile_size, tile_size, C)
    return tiles

# classify each tile as herbaceous or woody based on NDVI threshold of 0.55
def classify_tiles(tiles, ndvi_threshold=0.6):
    classifications = []
    for tile in tiles:
        # Assuming NDVI is in the first channel (or calculate it if needed)
        ndvi = tile[:, :, 0]  # Placeholder: replace with actual NDVI calculation if needed
        mean_ndvi = np.mean(ndvi)
        classification = 'herbaceous' if mean_ndvi < ndvi_threshold else 'woody'
        classifications.append(classification)
    return classifications

# create a pixel-level mask for the crown based on NDVI thresholding
# 0.55 gives is good, but keeps too many shadows
def crown_pixel_mask_from_ndvi_tiles(ndvi_tiles, threshold=0.6):
    """
    ndvi_tiles: (n_rows, n_cols, tile_h, tile_w, 1) or (n_rows, n_cols, tile_h, tile_w)
    returns: bool mask of same spatial shape (no channel)
    """
    if ndvi_tiles.ndim == 5:  # (n_rows, n_cols, H, W, 1)
        ndvi_tiles = ndvi_tiles[..., 0]

    crown_mask = ndvi_tiles > threshold
    return crown_mask


# play with min_pixels to make sure small trees are not missed. add max_pixels when largest crown is determined
def label_crowns_in_ndvi_tile(crown_mask, min_pixels=10):
    """
    crown_mask: (tile_h, tile_w) bool
    returns: labels (int32), regionprops list
    """
    # There are only two levels of connectivity with this, so try another method next
    labels = label(crown_mask, connectivity=1) 
    props = regionprops(labels)

    # Filter small blobs
    for region in props:
        if region.area < min_pixels:
            labels[labels == region.label] = 0

    return labels, props

def label_crowns_in_ndvi_tiles(crown_mask_tiles, min_pixels=20):
    n_rows, n_cols, H, W = crown_mask_tiles.shape
    labeled_tiles = np.zeros((n_rows, n_cols, H, W), dtype=np.int32)

    for i in range(n_rows):
        for j in range(n_cols):
            labels_ij, _ = label_crowns_in_ndvi_tile(
                crown_mask_tiles[i, j], min_pixels=min_pixels
            )
            labeled_tiles[i, j] = labels_ij

    return labeled_tiles



def show_labeled_tile(ndvi_tile, labels_tile, vmin=-1, vmax=1):
    # ndvi_tile: (H, W) or (H, W, 1)
    if ndvi_tile.ndim == 3:
        ndvi_tile = ndvi_tile[..., 0]

    # Turn labels into RGB overlay
    overlay = label2rgb(labels_tile, image=ndvi_tile,
                        bg_label=0, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ndvi_tile, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0].set_title("NDVI")
    axes[0].axis("off")

    axes[1].imshow(labels_tile, cmap="nipy_spectral")
    axes[1].set_title("Labels")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# to put the tiles back together
def tile_image(image, tile_size=512, overlap=64):
    H, W, C = image.shape
    step = tile_size - overlap

    tiles = []
    coords = []

    for y in range(0, H - tile_size + 1, step):
        for x in range(0, W - tile_size + 1, step):
            tiles.append(image[y:y+tile_size, x:x+tile_size, :])
            coords.append((y, x))   # top-left in original image

    return tiles, coords

# if the image is smaller than the tile size, we need to pad it first
def tile_image_with_padding(image, tile_size=512, overlap=64, pad_value=0):
    H, W, C = image.shape
    step = tile_size - overlap

    pad_h = (step - (H - tile_size) % step) % step if H > tile_size else tile_size - H
    pad_w = (step - (W - tile_size) % step) % step if W > tile_size else tile_size - W

    image_padded = np.pad(image,
                          ((0, pad_h), (0, pad_w), (0, 0)),
                          mode='constant', constant_values=pad_value)

    tiles, coords = tile_image(image_padded, tile_size, overlap)
    return tiles, coords, image_padded.shape

# for NDVI, try less than 0.55 being herbaceous, and above being woody. Based on classifying in QGIS

path = 'data/raw/20250728-ISA-NDVI-10cm.tif'
corkImg = load_image(path)
print(f"Loaded image shape: {corkImg.shape}")

ndvi_tiles = preprocess_image(corkImg, tile_size=512, overlap=64)
print(f"Preprocessed tiles shape: {ndvi_tiles.shape}")


crown_mask_tiles = crown_pixel_mask_from_ndvi_tiles(ndvi_tiles, threshold=0.55)
print(f"Crown mask tiles shape: {crown_mask_tiles.shape}")

labeled_crown_tiles = label_crowns_in_ndvi_tiles(crown_mask_tiles, min_pixels=20)
print(f"Labeled crown tiles shape: {labeled_crown_tiles.shape}")

for i in range(0, min(3, labeled_crown_tiles.shape[0])):
    for j in range(0, min(3, labeled_crown_tiles.shape[1])):
        print(f"Tile ({i}, {j})")
        show_labeled_tile(ndvi_tiles[i, j], labeled_crown_tiles[i, j])