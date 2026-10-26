from black.handle_ipynb_magics import TRANSFORMED_MAGICS
import numpy as np
import geopandas as gpd
import rasterio
from skimage.segmentation import watershed
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage


# TREE_TOPS = gpd.read_file("data/processed/20250728-ISA-tree-tops.tif")
with rasterio.open(
    "data/processed/20250728-ISA-tree-tops.tif"
    ) as src:
        marker_raster = src.read(1).astype("int32")
        marker_transform = src.transform
        marker_crs = src.crs
        marker_shape = marker_raster.shape
        treetop_binary = src.read(1) > 0
with rasterio.open("data/processed/20250728-ISA-chm-10cm.tif") as src:
    CHM = src.read(1)
    PROFILE = src.profile
    RASTER_CRS = src.crs
    TRANSFORM = src.transform
with rasterio.open("data/processed/chm_mask_over2m.tif") as src:
    CHM_MASK_2M = src.read(1).astype("bool")


binary_markers = marker_raster > 0

markers, n_markers = ndimage.label(
    binary_markers,
    structure=np.ones((3, 3), dtype=np.uint8)
)

markers = markers.astype("int32")

print("Number of marker regions:", n_markers)
print("Marker IDs:", np.unique(markers)[:20])


# not tested
def vegetation_height_mask(min_height, chm_smooth):
    min_height = 2.0  # metres

    vegetation_mask = (
        np.isfinite(chm_smooth) &
        (chm_smooth >= min_height)
    )
    return vegetation_mask

# 
def align_tree_points():
    with rasterio.open("data/processed/20250728-ISA-chm-10cm.tif") as src:
        chm = src.read(1)
        profile = src.profile
        transform = src.transform
        raster_crs = RASTER_CRS

    trees = markers.to_crs(raster_crs)
    return trees, chm, profile, transform


def build_watershed():
    surface = -CHM

    labels = watershed(
        surface,
        markers=markers,
        mask=CHM_MASK_2M,
        connectivity=2
    )
    return labels

# build_watershed()



def convert_labels_to_polygons(labels):
    crown_records = []

    for geom, value in shapes(
        labels.astype("int32"),
        mask=labels > 0,
        transform=TRANSFORM
    ):
        crown_records.append({
            "tree_id": int(value),
            "geometry": shape(geom)
        })

    crowns = gpd.GeoDataFrame(
        crown_records,
        crs=RASTER_CRS
    )
    return crowns

# run watershed
labels = watershed(
    -CHM,
    markers=markers,
    mask=CHM_MASK_2M,
    connectivity=2
)

# save result
output_profile = PROFILE.copy()
output_profile.update(
    dtype="int32",
    count=1,
    nodata=0,
    compress="lzw"
)

with rasterio.open(
    "crown_labels.tif",
    "w",
    **output_profile
) as dst:
    dst.write(labels.astype("int32"), 1)


# polygonize and save result
import rasterio.features
from shapely.geometry import shape as shapely_shape

records = []

for geometry, value in rasterio.features.shapes(
    labels.astype("int32"),
    mask=(labels > 0),
    transform=TRANSFORM,
    connectivity=8
):
    records.append({
        "tree_id": int(value),
        "geometry": shapely_shape(geometry)
    })

crowns = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=RASTER_CRS
)

crowns["area_m2"] = crowns.geometry.area

crowns.to_file(
    "tree_crowns_02.gpkg",
    layer="crowns",
    driver="GPKG"
)


# unique_markers = np.unique(markers)

# print("Unique marker values:", unique_markers[:20])
# print("Number of unique values:", len(unique_markers))
# print("Number of nonzero pixels:", np.count_nonzero(markers))
# print("Maximum marker value:", markers.max())

# if __name__ == "__main__":
#     # trees, chm, profile, transform = align_tree_points()
#     labels = build_watershed()
#     crowns = convert_labels_to_polygons(labels)
#     crowns.to_file("data/processed/tree_crowns.gpkg", driver="GPKG")