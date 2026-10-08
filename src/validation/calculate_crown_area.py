import geopandas as gpd
from pathlib import Path

import geopandas as gpd

total = 0

def print_total_polygon_area(gpkg_path, plot, layer=None, target_crs="EPSG:3763"):
    """
    Print the total area of all polygons in a GeoPackage.

    Parameters
    ----------
    gpkg_path : str
        Path to the .gpkg file.
    layer : str, optional
        Layer name. If None, geopandas reads the first layer.
    target_crs : str, optional
        Projected CRS used for area calculation.
    """
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    total_area_m2 = gdf.geometry.area.sum()

    print(f"{plot} Total area: {total_area_m2:.2f} m²")
    return total_area_m2

plot_names = ["plot61", "plot175", "plot238"]
pred_dir = Path("src/validation/yolo_plots")


for plot in plot_names:
    # gpkg_path = f"data/plots/{plot}/{plot}_crowns.gpkg"
    gpkg_path = pred_dir / f"{plot}.gpkg"
    total_area = print_total_polygon_area(gpkg_path, plot)
    total += total_area
print(total)