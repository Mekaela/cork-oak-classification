import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.plot import show

def save_array_as_raster(array, reference_tif, out_tif):
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=0
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(array.astype(rasterio.float32), 1)

def load_chm(chm_path):
    with rasterio.open(chm_path) as src:
        chm = src.read(1)  # first band
        transform = src.transform
        profile = src.profile
    return chm, transform, profile

def smooth_and_mask_chm(chm, height_threshold=2.0, sigma=1.0):
    chm_smooth = gaussian_filter(chm, sigma=sigma)  # reduce noise
    chm_masked = chm_smooth.copy()
    chm_masked[chm_smooth < height_threshold] = 0  # remove shrubs/ground (< threshold)
    return chm_masked

def extract_tree_tops(chm_masked):
    peaks = peak_local_max(chm_masked, min_distance=30, threshold_abs=1.0)
    tree_tops = np.zeros_like(chm_masked)
    tree_tops[peaks[:,0], peaks[:,1]] = chm_masked[peaks[:,0], peaks[:,1]]
    return tree_tops

load_chm_path = "data/processed/20250728-ISA-chm-10cm.tif"
chm, transform, profile = load_chm(load_chm_path)
chm_masked = smooth_and_mask_chm(chm, height_threshold=2.0, sigma=1.0)
tree_tops = extract_tree_tops(chm_masked)

save_array_as_raster(tree_tops, load_chm_path, "data/processed/20250728-ISA-tree-tops.tif")
# save_vector_preview(tree_tops, "data/processed/20250728-ISA-tree-tops.png", raster_path=load_chm_path)