from pathlib import Path
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.windows import Window

SITE_PATH = "data/raw"
IMAGE = SITE_PATH + "/training_clipped_RGB.tif"
GPKG = SITE_PATH + "/clipped_training_polygons_60ha.gpkg"
GPKG_LAYER = "clipped"


OUT = Path("datasets_site")

TILE_SIZE = 640
OVERLAP = 100
STEP = TILE_SIZE - OVERLAP

VAL_FRACTION = 0.20
MIN_FRACTION = 0.10
CLASS_ID = 0


def tile_split(x_start, image_width, val_fraction=0.20):
    """
    Put a spatially separate eastern strip into validation.

    Tile assignment uses the tile centre, rather than polygon centroids,
    so one tile cannot be written into both train and val.
    """
    tile_center_x = x_start + TILE_SIZE / 2
    val_start_x = image_width * (1 - val_fraction)

    return "val" if tile_center_x >= val_start_x else "train"


def polygon_to_yolo(geom, transform, tile_x, tile_y, tile_w, tile_h, class_id=0):
    """Convert one clipped map-coordinate polygon to a YOLO segment row."""
    if geom.is_empty:
        return None

    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)

    if geom.geom_type != "Polygon":
        return None

    coords = np.asarray(geom.exterior.coords)

    # Map x/y -> full raster row/column. rowcol returns rows, cols.
    rows, cols = rasterio.transform.rowcol(
        transform,
        coords[:, 0],
        coords[:, 1],
    )

    local_x = np.asarray(cols, dtype=float) - tile_x
    local_y = np.asarray(rows, dtype=float) - tile_y

    # Reject malformed geometry instead of distorting it with clipping.
    if (
        np.any(local_x < -1)
        or np.any(local_x > tile_w + 1)
        or np.any(local_y < -1)
        or np.any(local_y > tile_h + 1)
    ):
        return None

    x_norm = np.clip(local_x / tile_w, 0, 1)
    y_norm = np.clip(local_y / tile_h, 0, 1)

    # Drop the repeated final point, if the ring is explicitly closed.
    points = np.column_stack((x_norm, y_norm))
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    if len(points) < 3:
        return None

    flattened = " ".join(f"{value:.6f}" for value in points.ravel())
    return f"{class_id} {flattened}"


# Create all dataset directories.
for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

with rasterio.open(IMAGE) as src:
    crowns = gpd.read_file(GPKG, layer=GPKG_LAYER)

    if crowns.crs != src.crs:
        raise ValueError(
            f"CRS mismatch: crowns={crowns.crs}, image={src.crs}"
        )

    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty].copy()
    crowns = crowns[crowns.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if crowns.empty:
        raise ValueError("No Polygon or MultiPolygon crowns found in the GeoPackage.")

    sindex = crowns.sindex
    tile_rows = []
    tile_id = 0

    for tile_y in range(0, src.height, STEP):
        for tile_x in range(0, src.width, STEP):
            tile_w = min(TILE_SIZE, src.width - tile_x)
            tile_h = min(TILE_SIZE, src.height - tile_y)

            if tile_w < TILE_SIZE or tile_h < TILE_SIZE:
                continue  # Keep fixed-size tiles only for this first dataset.

            # src.window_bounds returns map bounds in image CRS.
            window = Window(tile_x, tile_y, tile_w, tile_h)
            left, bottom, right, top = src.window_bounds(window)
            tile_geom = box(left, bottom, right, top)

            candidate_idx = list(sindex.query(tile_geom, predicate="intersects"))
            if not candidate_idx:
                continue  # Change later if you intentionally want empty/background tiles.

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
                candidates["fraction_retained"] >= MIN_FRACTION
            ].copy()

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
                    class_id=CLASS_ID,
                )
                if label is not None:
                    label_rows.append(label)

            if not label_rows:
                continue

            split = tile_split(tile_x, src.width, VAL_FRACTION)

            # Read and save a true RGB tile.
            rgb = src.read([1, 2, 3], window=window)
            rgb = np.moveaxis(rgb, 0, -1)

            if rgb.dtype != np.uint8:
                band_min = rgb.min(axis=(0, 1), keepdims=True)
                band_max = rgb.max(axis=(0, 1), keepdims=True)
                rgb = (
                    255 * (rgb - band_min) / np.maximum(band_max - band_min, 1)
                ).astype(np.uint8)

            image_name = f"tile_{tile_id:06d}_x{tile_x}_y{tile_y}.jpg"
            label_name = image_name.replace(".jpg", ".txt")

            # cv2.imwrite expects BGR pixels.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(OUT / "images" / split / image_name),
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            (OUT / "labels" / split / label_name).write_text(
                "\n".join(label_rows) + "\n"
            )

            tile_rows.append({
                "tile_id": tile_id,
                "split": split,
                "filename": image_name,
                "x_start": tile_x,
                "y_start": tile_y,
                "width": tile_w,
                "height": tile_h,
                "instances": len(label_rows),
            })

            tile_id += 1

with open(OUT / "tile_index.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=tile_rows[0].keys())
    writer.writeheader()
    writer.writerows(tile_rows)

print(f"Created {len(tile_rows)} annotated tiles.")
print(f"Train: {sum(row['split'] == 'train' for row in tile_rows)}")
print(f"Val:   {sum(row['split'] == 'val' for row in tile_rows)}")