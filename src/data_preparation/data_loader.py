# Load and preprocess images (rasterio, opencv)
import numpy as np
import rasterio
from skimage.util import view_as_windows
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import features


def load_tif_with_mask(path):
    with rasterio.open(path) as src:
        data = src.read(masked=True)  # (bands, H, W)

        # Valid where every band is valid.
        mask = ~np.any(data.mask, axis=0)

        # Convert masked cells to NaN instead of exposing fill values.
        dsm = data.filled(np.nan)

        # Convert from (bands, H, W) to (H, W, bands)
        dsm = np.transpose(dsm, (1, 2, 0)).astype(np.float32)

        profile = src.profile.copy()

    return dsm, mask, profile


def preprocess_image(image, tile_size=512, overlap=64):
    step = tile_size - overlap
    # image: (H, W, C)
    tiles = view_as_windows(image, (tile_size, tile_size, image.shape[2]), step=step)
    # tiles shape: (n_rows, n_cols, 1, tile_size, tile_size, C)
    tiles = tiles.squeeze(axis=2)  
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


def tif_to_polygon(input_tif, output_geojson="boundary.geojson"):
    try:
        dsm, mask, profile = load_tif_with_mask(input_tif)
        
        print(f"Image shape: {dsm.shape}")
        print(f"Valid pixels: {np.count_nonzero(mask)}")

        # Extract polygon geometries from the boolean mask
        # We transform pixel coordinates to real-world coordinates using the profile transform
        results = features.shapes(
            mask.astype(np.uint8), 
            mask=mask, 
            transform=profile['transform']
        )

        polygons = []
        for geom, value in results:
            polygons.append(Polygon(geom['coordinates'][0]))

        if not polygons:
            raise ValueError("No valid polygons found. The mask might be empty or all False.")

        # (Optional) Keep only the largest polygon
        largest_poly = max(polygons, key=lambda a: a.area)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [{"id": 1, "geometry": largest_poly}], 
            crs=profile['crs']
        )

        # Save to GeoJSON
        gdf.to_file(output_geojson, driver="GeoJSON")
        
        print(f"Success! Boundary saved to {output_geojson}")
        print(f"Boundary Area: {largest_poly.area:.2f} square units")

    except Exception as e:
        print(f"Error generating polygon: {e}")

# tif_to_polygon('data/raw/20250728-ISA-RGB-10cm.tif')