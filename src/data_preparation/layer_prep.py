import rasterio
from scipy.ndimage import grey_erosion, binary_erosion
import numpy as np
from pathlib import Path
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import Window
from src.data_preparation.data_loader import load_tif_with_mask

# ------build CHM from DSM---------
def estimate_dtm_from_dsm(
    dsm,
    valid_mask,
    window_size=201
):
    if dsm.ndim != 3 or dsm.shape[2] != 1:
        raise ValueError("Expected dsm shape (H, W, 1)")

    if valid_mask.shape != dsm.shape[:2]:
        raise ValueError("valid_mask must have shape (H, W)")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    dsm_float = dsm.astype(np.float32, copy=False)

    valid = (valid_mask & np.isfinite(dsm_float[..., 0]))

    # Invalid pixels cannot become the local minimum.
    filtered = np.where(valid[..., np.newaxis], dsm_float, np.inf)

    dtm = grey_erosion(filtered, size=(window_size, window_size, 1), mode="constant", cval=np.inf, origin=0)

    dtm[..., 0][~valid] = np.nan

    return dtm

def compute_chm_from_dsm(dsm, dtm, mask, nodata=0.0):
    chm = dsm - dtm                      # (H, W, 1)

    # Apply polygon mask: keep only inside, set outside to nodata
    chm_out = chm.copy()
    chm_out[~mask, 0] = nodata          # mask is (H, W), chm_out is (H, W, 1)

    edge_margin = 100   # smaller amounts were not enough
    # inner_mask = binary_erosion(mask, iterations=edge_margin)
    inner_mask = binary_erosion(mask)
    chm_display = chm_out.copy()
    chm_display[~inner_mask, 0] = np.nan  # hide outer band

    return chm_display

def save_layer(chm, profile, out_path, nodata=0.0):
    # chm: (H, W, 1)
    out = chm[:, :, 0].astype(np.float32)    # single band

    prof = profile.copy()
    prof.update(
        count=1,
        dtype="float32",
        nodata=nodata
    )

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)

def build_chm_mask(chm):
    return (chm[:, :, 0] > 2.0).astype(np.uint8)

# to save the CHM:
# save_chm(chm, profile, "20250728-ISA-CHM-10cm.tif", nodata=0.0)

# --------------------------  

# ---------change 4 band RGB to 3 band RGB for model training----------------

train_sites = ["plot10", "plot14", "plot39", "plot43", "plot88", "plot143", "plot183", "plot205", "plot242"]
val_sites = ["plot61", "plot175", "plot238"]

# replace as 3 band RGB
def four_to_three_band(src_path, dst_path):
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=3)

        rgb = src.read([1, 2, 3])
 
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(rgb)






def build_output_dir(base_dir="data", prefix="yolo_sites_data"):
    """Create a date-stamped output directory for the YOLO dataset."""
    today = datetime.now().strftime("%Y%m%d")
    return Path(base_dir) / f"{prefix}_{today}"


def create_dataset_directories(output_dir):
    """Create the standard YOLO images/labels directory structure."""
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_crowns(gpkg_path, layer):
    """Load valid Polygon or MultiPolygon crowns from a GeoPackage."""
    crowns = gpd.read_file(gpkg_path, layer=layer)

    crowns = crowns[
        crowns.geometry.notna()
        & ~crowns.geometry.is_empty
    ].copy()

    crowns = crowns[
        crowns.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if crowns.empty:
        raise ValueError(
            "No Polygon or MultiPolygon crowns found in the GeoPackage."
        )

    return crowns


def validate_crs(crowns, raster_src):
    """Ensure the crown polygons and raster use the same CRS."""
    if crowns.crs != raster_src.crs:
        raise ValueError(
            f"CRS mismatch: crowns={crowns.crs}, image={raster_src.crs}"
        )


def tile_split(x_start, image_width, tile_size, val_fraction=0.20):
    """
    Assign a tile to train or validation using a spatially separate
    eastern strip. The tile centre determines the split.
    """
    tile_center_x = x_start + tile_size / 2
    val_start_x = image_width * (1 - val_fraction)

    return "val" if tile_center_x >= val_start_x else "train"


def polygon_to_yolo(
    geom,
    transform,
    tile_x,
    tile_y,
    tile_w,
    tile_h,
    class_id=0,
):
    """Convert one clipped map-coordinate polygon to a YOLO segment row."""
    if geom.is_empty:
        return None

    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)

    if geom.geom_type != "Polygon":
        return None

    coords = np.asarray(geom.exterior.coords)

    rows, cols = rasterio.transform.rowcol(
        transform,
        coords[:, 0],
        coords[:, 1],
    )

    local_x = np.asarray(cols, dtype=float) - tile_x
    local_y = np.asarray(rows, dtype=float) - tile_y

    outside_tolerance = 1
    if (
        np.any(local_x < -outside_tolerance)
        or np.any(local_x > tile_w + outside_tolerance)
        or np.any(local_y < -outside_tolerance)
        or np.any(local_y > tile_h + outside_tolerance)
    ):
        return None

    x_norm = np.clip(local_x / tile_w, 0, 1)
    y_norm = np.clip(local_y / tile_h, 0, 1)

    points = np.column_stack((x_norm, y_norm))

    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    if len(points) < 3:
        return None

    flattened = " ".join(f"{value:.6f}" for value in points.ravel())
    return f"{class_id} {flattened}"


def get_crowns_for_tile(crowns, tile_geom, min_fraction=0.10):
    """
    Find crowns intersecting a tile, clip them to the tile, and retain
    only those whose retained area meets the minimum fraction.
    """
    sindex = crowns.sindex
    candidate_idx = list(sindex.query(tile_geom, predicate="intersects"))

    if not candidate_idx:
        return gpd.GeoDataFrame(columns=crowns.columns, crs=crowns.crs)

    candidates = crowns.iloc[candidate_idx].copy()
    candidates["original_area"] = candidates.geometry.area
    candidates["geometry"] = candidates.geometry.intersection(tile_geom)

    candidates = candidates[
        candidates.geometry.notna()
        & ~candidates.geometry.is_empty
    ].copy()

    candidates["fraction_retained"] = (
        candidates.geometry.area / candidates["original_area"]
    )

    candidates = candidates[
        candidates["fraction_retained"] >= min_fraction
    ].copy()

    return candidates


def normalise_to_uint8(rgb):
    """Convert an RGB array to uint8, scaling only if necessary."""
    if rgb.dtype == np.uint8:
        return rgb

    band_min = rgb.min(axis=(0, 1), keepdims=True)
    band_max = rgb.max(axis=(0, 1), keepdims=True)

    return (
        255
        * (rgb - band_min)
        / np.maximum(band_max - band_min, 1)
    ).astype(np.uint8)


def save_tile_and_labels(
    rgb,
    label_rows,
    output_dir,
    split,
    tile_id,
    tile_x,
    tile_y,
):
    """Write one RGB JPEG tile and its YOLO segmentation label file."""
    image_name = f"tile_{tile_id:06d}_x{tile_x}_y{tile_y}.jpg"
    label_name = image_name.replace(".jpg", ".txt")

    rgb = normalise_to_uint8(rgb)

    # OpenCV expects BGR rather than RGB.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        str(output_dir / "images" / split / image_name),
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    (output_dir / "labels" / split / label_name).write_text(
        "\n".join(label_rows) + "\n"
    )

    return image_name


def create_yolo_dataset(
    image_path,
    gpkg_path,
    gpkg_layer,
    output_dir,
    tile_size=640,
    overlap=100,
    val_fraction=0.20,
    min_fraction=0.10,
    class_id=0,
):
    """
    Create a YOLO segmentation dataset from an RGB raster and crown
    polygons. Only tiles containing valid retained crown geometry are saved.
    """
    step = tile_size - overlap

    create_dataset_directories(output_dir)

    tile_rows = []
    tile_id = 0

    with rasterio.open(image_path) as src:
        crowns = load_crowns(gpkg_path, gpkg_layer)
        validate_crs(crowns, src)

        for tile_y in range(0, src.height, step):
            for tile_x in range(0, src.width, step):
                tile_w = min(tile_size, src.width - tile_x)
                tile_h = min(tile_size, src.height - tile_y)

                # Keep fixed-size tiles only for this initial dataset.
                if tile_w < tile_size or tile_h < tile_size:
                    continue

                window = Window(tile_x, tile_y, tile_w, tile_h)
                left, bottom, right, top = src.window_bounds(window)
                tile_geom = box(left, bottom, right, top)

                candidates = get_crowns_for_tile(
                    crowns,
                    tile_geom,
                    min_fraction=min_fraction,
                )

                if candidates.empty:
                    continue

                label_rows = []

                for geom in candidates.geometry:
                    label = polygon_to_yolo(
                        geom=geom,
                        transform=src.transform,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        tile_w=tile_w,
                        tile_h=tile_h,
                        class_id=class_id,
                    )

                    if label is not None:
                        label_rows.append(label)

                if not label_rows:
                    continue

                split = tile_split(
                    x_start=tile_x,
                    image_width=src.width,
                    tile_size=tile_size,
                    val_fraction=val_fraction,
                )

                rgb = src.read([1, 2, 3], window=window)
                rgb = np.moveaxis(rgb, 0, -1)

                image_name = save_tile_and_labels(
                    rgb=rgb,
                    label_rows=label_rows,
                    output_dir=output_dir,
                    split=split,
                    tile_id=tile_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )

                tile_rows.append(
                    {
                        "tile_id": tile_id,
                        "split": split,
                        "filename": image_name,
                        "x_start": tile_x,
                        "y_start": tile_y,
                        "width": tile_w,
                        "height": tile_h,
                        "instances": len(label_rows),
                    }
                )

                tile_id += 1

    print(f"Created {len(tile_rows)} annotated tiles.")
    print(f"Train: {sum(row['split'] == 'train' for row in tile_rows)}")
    print(f"Val:   {sum(row['split'] == 'val' for row in tile_rows)}")

    return tile_rows


# ----to build CHM and CHM 2m mask------
dsm_path = 'data/raw/20250728-ISA-dsm-10cm.tif'
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



# -------to make 4 band to 3 band------
# if modifying one large image:
# src_path = f"data/raw/20250728_ISA_RGB_10cm.tif"
# dst_path = f"data/plots/20250728_ISA_RGB_3band.tif"
# four_to_three_band(src_path, dst_path)

# if modifying plots:
# for site in train_sites:
#     src_path = f"data/plots/{site}/{site}_RGB.tif"
#     dst_path = f"data/plots/{site}/{site}_RGB_3band.tif"
#     four_to_three_band(src_path, dst_path)


# -----to create the YOLO train and val dataset from the RGB and crowns-----
today = datetime.now().strftime("%Y%m%d")

site_path = Path("data/raw")
# this should be the 3 band RGB image, not the 4 band one, in this case the training plots have been manually removed
# where this manual removal is not needed, the image path can be the dst path from above
image_path = site_path / "training_clipped_RGB.tif"
gpkg_path = site_path / "clipped_training_polygons_60ha.gpkg"
gpkg_layer = "clipped"

output_dir = Path(f"data/yolo_sites_data_{today}")

create_yolo_dataset(
    image_path=image_path,
    gpkg_path=gpkg_path,
    gpkg_layer=gpkg_layer,
    output_dir=output_dir,
    tile_size=640,
    overlap=100,
    val_fraction=0.20,
    min_fraction=0.10,
    class_id=0,
)

