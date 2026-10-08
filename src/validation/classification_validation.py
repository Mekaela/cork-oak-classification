
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from pathlib import Path


TREE = 1
NON_TREE = 0
BINARY_LABELS = [NON_TREE, TREE]


# --------------------------------------------------------------
# Raster grid and vector handling
# --------------------------------------------------------------
def get_raster_grid(grid_raster_path):
    """Read the CRS, transform, dimensions, and bounds of a grid raster."""
    with rasterio.open(grid_raster_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def read_crown_polygons(vector_path, grid):
    """Read crown polygons and reproject them to the grid CRS if needed."""
    gdf = gpd.read_file(vector_path)

    if gdf.crs is None:
        raise ValueError(f"No CRS defined for: {vector_path}")

    if gdf.crs != grid["crs"]:
        gdf = gdf.to_crs(grid["crs"])

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    if gdf.empty:
        raise ValueError(f"No valid crown polygons found in: {vector_path}")

    return gdf


def rasterize_tree_mask(gdf, grid):
    """
    Convert crown polygons into a binary tree mask.

    1 = tree pixel
    0 = non-tree pixel
    """
    shapes = ((geom, TREE) for geom in gdf.geometry)

    return rasterize(
        shapes=shapes,
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=NON_TREE,
        all_touched=False,
        dtype="uint8",
    )


# --------------------------------------------------------------
# Accuracy calculation
# --------------------------------------------------------------
def calculate_binary_accuracy(y_true, y_pred):
    """
    Calculate overall accuracy and Cohen's kappa for binary tree masks.

    y_true and y_pred must have identical shapes and contain 0 or 1.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Reference and predicted masks must have identical dimensions."
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
    )

    tn, fp, fn, tp = cm.ravel()
    n_pixels = tp + tn + fp + fn

    overall_accuracy = (tp + tn) / n_pixels

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
    )

    return {
        "plot_id": None,
        "n_pixels": n_pixels,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "overall_accuracy": overall_accuracy,
        "kappa": kappa,
        "confusion_matrix": cm,
    }

def check_file_exists(path, description):
    """Raise a clear error if an expected input file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {description} for plot:\n{path}"
        )
# --------------------------------------------------------------
# Plot evaluation
# --------------------------------------------------------------
def evaluate_plot(plot_paths):
    """Evaluate one plot using formatted reference/predicted/grid paths."""
    plot_id = plot_paths["plot_id"]

    """Raise a clear error if an expected input file is missing."""
    if not Path(plot_paths["reference_vector"]).exists():
        raise FileNotFoundError(
            f"Missing {plot_id} reference crowns for plot"
        )

    if not Path(plot_paths["predicted_vector"]).exists():
            raise FileNotFoundError(
                f"Missing {plot_id} predicted crowns for plot"
            )
    
    if not Path(plot_paths["grid_raster"]).exists():
            raise FileNotFoundError(
                f"Missing {plot_id} raster for plot"
            )

    grid = get_raster_grid(plot_paths["grid_raster"])

    reference_gdf = read_crown_polygons(
        plot_paths["reference_vector"],
        grid,
    )

    predicted_gdf = read_crown_polygons(
        plot_paths["predicted_vector"],
        grid,
    )

    reference_mask = rasterize_tree_mask(reference_gdf, grid)
    predicted_mask = rasterize_tree_mask(predicted_gdf, grid)

    results = calculate_binary_accuracy(
        y_true=reference_mask,
        y_pred=predicted_mask,
    )

    results["plot_id"] = plot_id
    results["reference_vector"] = str(plot_paths["reference_vector"])
    results["predicted_vector"] = str(plot_paths["predicted_vector"])
    results["grid_raster"] = str(plot_paths["grid_raster"])

    return results


def save_binary_raster(mask, grid, output_path):
    """Save a binary 0/1 tree mask as a GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid["height"],
        width=grid["width"],
        count=1,
        dtype="uint8",
        crs=grid["crs"],
        transform=grid["transform"],
        nodata=None,
    ) as dst:
        dst.write(mask, 1)


# --------------------------------------------------------------
# Run all plots
# --------------------------------------------------------------
def evaluate_all_plots(plot_names, pred_dir, ref_dir):
    """Loop through plot names, build paths, and calculate accuracy."""
    results = []

    for plot_id in plot_names:
        print(f"Evaluating: {plot_id}")

        plot_paths = {
            "plot_id": plot_id,
            "reference_vector": (ref_dir / plot_id / f"{plot_id}_crowns.gpkg"),
            "predicted_vector": (pred_dir / f"{plot_id}.gpkg"),
            "grid_raster": (pred_dir / f"{plot_id}_classified_01.tif"),
        }

        try:
            plot_result = evaluate_plot(
                plot_paths,
            )

        except Exception as exc:
            print(f"FAILED: {plot_id}\n{exc}\n")
            continue

        results.append(plot_result)
        print(
            f"Completed {plot_id}: "
            f"OA = {plot_result['overall_accuracy']:.4f}, "
            f"kappa = {plot_result['kappa']:.4f}\n"
        )

    if not results:
        raise RuntimeError("No plots were successfully evaluated.")

    results_df = pd.DataFrame(results)

    output_columns = [
        "plot_id",
        "n_pixels",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "overall_accuracy",
        "kappa",
        "reference_vector",
        "predicted_vector",
        # "grid_raster",
    ]

    results_df = results_df[output_columns]
    print("\nPer-plot results:")
    print(results_df)

    return results_df


# --------------------------------------------------------------
# Pooled accuracy across all plots
# --------------------------------------------------------------
def calculate_pooled_accuracy(results_df):
    """
    Pool confusion-matrix counts across plots and recalculate metrics.

    This is preferable to averaging per-plot accuracies because plots
    with more pixels contribute proportionally more.
    """
    tp = int(results_df["true_positives"].sum())
    tn = int(results_df["true_negatives"].sum())
    fp = int(results_df["false_positives"].sum())
    fn = int(results_df["false_negatives"].sum())

    y_true = (
        [NON_TREE] * (tn + fp)
        + [TREE] * (fn + tp)
    )

    y_pred = (
        [NON_TREE] * tn
        + [TREE] * fp
        + [NON_TREE] * fn
        + [TREE] * tp
    )

    pooled = calculate_binary_accuracy(y_true, y_pred)
    print("\nPooled results across all plots:")
    print(f"Total pixels: {pooled['n_pixels']}")
    print(f"Overall accuracy: {pooled['overall_accuracy']:.4f}")
    print(f"Cohen's kappa: {pooled['kappa']:.4f}")
    print("\nConfusion matrix:")
    print(pooled["confusion_matrix"])

    return pooled

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


