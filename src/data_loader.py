# Load and preprocess images (rasterio, opencv)
import os
import numpy as np
import rasterio
from skimage.util import view_as_windows
# import cv2  
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def load_dsm_with_mask(path):
    with rasterio.open(path) as src:
        # Read as masked array so nodata and internal .msk are honored
        data = src.read(masked=True)       # (bands, H, W), masked
        dsm = np.transpose(data.data, (1, 2, 0))  # (H, W, C)
        # valid where not masked
        mask = ~data.mask[0]              # (H, W), assume shared mask across bands
        profile = src.profile
    return dsm, mask, profile

def preprocess_image(image, tile_size=512, overlap=64):
    step = tile_size - overlap
    # image: (H, W, C)
    tiles = view_as_windows(image, (tile_size, tile_size, image.shape[2]), step=step)
    # tiles shape: (n_rows, n_cols, 1, tile_size, tile_size, C)
    tiles = tiles.squeeze(axis=2)  # (n_rows, n_cols, tile_size, tile_size, C)
    return tiles

def reconstruct_image(tiles, image_shape, mask, tile_size=512, overlap=64):
    """
    tiles: (n_rows, n_cols, tile_h, tile_w, C) from preprocess_image(img, ...)
    image_shape: original (H, W, C)
    mask: (H, W) boolean, True inside polygon
    """
    H, W, C = image_shape
    step = tile_size - overlap
    n_rows, n_cols, tile_h, tile_w, _ = tiles.shape

    out = np.zeros((H, W, C), dtype=tiles.dtype)
    weight = np.zeros((H, W), dtype=np.float32)  # scalar weight per pixel

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * step
            x0 = j * step
            y1 = y0 + tile_h
            x1 = x0 + tile_w

            # restrict to image bounds
            y1 = min(y1, H)
            x1 = min(x1, W)

            tile = tiles[i, j, :y1 - y0, :x1 - x0, :]
            tile_mask = mask[y0:y1, x0:x1]          # boolean

            # expand mask to channels for broadcasting
            tm3 = tile_mask[..., None]

            # accumulate only valid (inside polygon) pixels
            out[y0:y1, x0:x1, :] += tile * tm3
            weight[y0:y1, x0:x1] += tile_mask.astype(np.float32)

    # normalize where we have at least one valid contribution
    valid = weight > 0
    out[valid, :] /= weight[valid, None]

    # outside polygon: set to nodata or keep as zero
    out[~mask, :] = 0  # or your DSM nodata value
    return out

# if padding is needed
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
