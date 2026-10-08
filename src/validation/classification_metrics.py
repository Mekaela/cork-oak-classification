
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from pathlib import Path

BASE_DIR = Path("src/validation/")
model = 'yolo_plots'

PLOTS = [
    "plot61",
    "plot175",
    "plot238",
]

REFERENCE_CROWNS_DIR = Path("data/plots")
PREDICTED_CROWNS_DIR = BASE_DIR / model

RESULTS_DIR = BASE_DIR / "accuracy_results"
MASK_DIR = RESULTS_DIR / "tree_masks"

REFERENCE_SUFFIX = "_crowns.gpkg"

ALL_TOUCHED = False


# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
TREE = 1
NON_TREE = 0
BINARY_LABELS = [NON_TREE, TREE]


# --------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------
def build_plot_paths(plot_id):
    """Create the expected input paths for one plot."""
    return {
        "plot_id": plot_id,
        "reference_vector": (
            REFERENCE_CROWNS_DIR
            / plot_id
            / f"{plot_id}{REFERENCE_SUFFIX}"
        ),
        "predicted_vector": (
            PREDICTED_CROWNS_DIR
            / f"{plot_id}.gpkg"
        ),
        "grid_raster": (
            PREDICTED_CROWNS_DIR
            / f"{plot_id}_classified_01.tif"
        ),
    }


def check_file_exists(path, description):
    """Raise a clear error if an expected input file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {description} for plot:\n{path}"
        )


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


def rasterize_tree_mask(gdf, grid, all_touched=ALL_TOUCHED):
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
        all_touched=all_touched,
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


# --------------------------------------------------------------
# Plot evaluation
# --------------------------------------------------------------
def evaluate_plot(plot_paths, save_masks=True):
    """Evaluate one plot using formatted reference/predicted/grid paths."""
    plot_id = plot_paths["plot_id"]

    check_file_exists(
        plot_paths["reference_vector"],
        f"{plot_id} reference crowns",
    )

    check_file_exists(
        plot_paths["predicted_vector"],
        f"{plot_id} predicted crowns",
    )

    check_file_exists(
        plot_paths["grid_raster"],
        f"{plot_id} grid raster",
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

    if save_masks:
        save_binary_raster(
            reference_mask,
            grid,
            MASK_DIR / f"{plot_id}_reference_tree_mask.tif",
        )

        save_binary_raster(
            predicted_mask,
            grid,
            MASK_DIR / f"{plot_id}_predicted_tree_mask.tif",
        )

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
def evaluate_all_plots(plot_names, save_masks=True):
    """Loop through plot names, build paths, and calculate accuracy."""
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for plot_id in plot_names:
        print(f"Evaluating: {plot_id}")

        plot_paths = build_plot_paths(plot_id)

        try:
            plot_result = evaluate_plot(
                plot_paths,
                save_masks=save_masks,
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
        "grid_raster",
    ]

    results_df = results_df[output_columns]
    results_csv = RESULTS_DIR / f"{model}_plot_accuracy_results.csv"
    results_df.to_csv(results_csv, index=False)

    return results_df, results_csv


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

    return pooled


# --------------------------------------------------------------
# Run script
# --------------------------------------------------------------
if __name__ == "__main__":
    results_df, results_csv = evaluate_all_plots(
        plot_names=PLOTS,
        save_masks=True,
    )

    print("\nPer-plot results:")
    print(results_df)

    pooled = calculate_pooled_accuracy(results_df)

    print("\nPooled results across all plots:")
    print(f"Total pixels: {pooled['n_pixels']}")
    print(f"Overall accuracy: {pooled['overall_accuracy']:.4f}")
    print(f"Cohen's kappa: {pooled['kappa']:.4f}")
    print("\nConfusion matrix:")
    print(pooled["confusion_matrix"])

    print(f"\nPer-plot results saved to:\n{results_csv}")