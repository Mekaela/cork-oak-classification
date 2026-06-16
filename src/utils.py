import geopandas as gpd
from shapely.geometry import Polygon, mapping
from rasterio import features
import numpy as np
from data_loader import load_tif_with_mask
# Helper functions (e.g., polygon to mask, metrics)

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