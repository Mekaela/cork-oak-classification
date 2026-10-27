import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from datetime import datetime

CHM_PATH = "data/processed/20250728-ISA-chm-10cm.tif"
MASK_PATH = "data/processed/chm_mask_over2m.tif"
OUTPUT_GPKG = f"tree_crowns__{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpkg"


def load_raster_band(path):
    """Load a single-band raster and its georeferencing."""
    with rasterio.open(path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile
    return band, transform, crs, profile


def smooth_and_mask_chm(chm, height_threshold=2.0, sigma=1.0):
    """Smooth the CHM and remove pixels below the height threshold."""
    chm_smooth = gaussian_filter(chm, sigma=sigma)
    chm_masked = chm_smooth.copy()
    chm_masked[chm_smooth < height_threshold] = 0
    return chm_masked


def extract_tree_tops(chm_masked, min_distance=30, threshold_abs=1.0):
    """Identify local maxima in the CHM as tree-top markers."""
    peaks = peak_local_max(
        chm_masked,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )

    tree_tops = np.zeros_like(chm_masked, dtype="int32")
    tree_tops[peaks[:, 0], peaks[:, 1]] = 1
    return tree_tops


def label_tree_tops(tree_tops):
    """Convert isolated tree-top pixels into connected marker regions."""
    markers, n_markers = ndimage.label(
        tree_tops > 0,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    return markers.astype("int32"), n_markers


def run_watershed(chm, markers, mask):
    """Segment tree crowns using the inverted CHM as the elevation surface."""
    return watershed(
        -chm,
        markers=markers,
        mask=mask,
        connectivity=2,
    )


def labels_to_crowns(labels, transform, crs):
    """Convert labelled crown pixels into a GeoDataFrame of polygons."""
    records = []

    for geom, value in shapes(
        labels.astype("int32"),
        mask=labels > 0,
        transform=transform,
        connectivity=8,
    ):
        records.append(
            {
                "tree_id": int(value),
                "geometry": shape(geom),
            }
        )

    crowns = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )

    crowns["area_m2"] = crowns.geometry.area
    return crowns


def save_crowns(crowns, output_path, layer="crowns"):
    """Save the final crown polygons. This is the only file-writing step."""
    crowns.to_file(
        output_path,
        layer=layer,
        driver="GPKG",
    )


def extract_tree_crowns(
    chm_path,
    mask_path,
    output_path,
    height_threshold=2.0,
    sigma=1.0,
    min_distance=30,
    peak_threshold=1.0,
):
    """Run the complete tree-crown segmentation workflow."""
    chm, transform, crs, profile = load_raster_band(chm_path)
    mask, _, _, _ = load_raster_band(mask_path)
    mask = mask.astype(bool)

    chm_masked = smooth_and_mask_chm(
        chm,
        height_threshold=height_threshold,
        sigma=sigma,
    )

    tree_tops = extract_tree_tops(
        chm_masked,
        min_distance=min_distance,
        threshold_abs=peak_threshold,
    )

    markers, n_markers = label_tree_tops(tree_tops)
    print(f"Detected {n_markers} tree-top marker regions.")

    labels = run_watershed(chm, markers, mask)
    crowns = labels_to_crowns(labels, transform, crs)

    save_crowns(crowns, output_path)
    print(f"Saved {len(crowns)} crown polygons to {output_path}.")

    return crowns


if __name__ == "__main__":
    extract_tree_crowns(
        chm_path=CHM_PATH,
        mask_path=MASK_PATH,
        output_path=OUTPUT_GPKG,
    )