import numpy as np
from skimage.util import view_as_windows
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt


def load_image(path):
    """
    Load a multi-band image from a file path.
    Returns the image as a NumPy array with shape (H, W, C).
    """
    import rasterio

    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, bands)
    return img

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
