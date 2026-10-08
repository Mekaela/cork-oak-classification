from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from scipy.optimize import linear_sum_assignment
from shapely.validation import make_valid
from datetime import date

model = "yolo_plots"
today = date.today()
pred_dir = Path(f"src/validation/{model}")
ref_dir = Path("data/plots")
chm_dir = Path("data/plots")

plot_names = ["plot61", "plot175", "plot238"]

MIN_IOU = 0.3

OUTPUT_CSV = f"{today}_{model}_matched_model_crown_heights.csv"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_crowns(field_path, model_path):
    field = gpd.read_file(field_path)
    model = gpd.read_file(model_path)

    field["geometry"] = field.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )
    model["geometry"] = model.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )

    field = field[field.geometry.notna() & ~field.geometry.is_empty].copy()
    model = model[model.geometry.notna() & ~model.geometry.is_empty].copy()

    if field.crs != model.crs:
        raise ValueError(
            f"CRS mismatch: field={field.crs}, model={model.crs}"
        )

    return field.reset_index(drop=True), model.reset_index(drop=True)


def compute_iou_matrix(field, model):
    iou = np.zeros((len(field), len(model)), dtype=float)

    for i, field_geom in enumerate(field.geometry.values):
        candidates = model.sindex.query(
            field_geom, predicate="intersects"
        )

        for j in candidates:
            model_geom = model.geometry.iloc[j]

            intersection_area = field_geom.intersection(model_geom).area
            union_area = field_geom.union(model_geom).area

            if union_area > 0:
                iou[i, j] = intersection_area / union_area

    return iou


def extract_model_chm_height(model, chm_path, stat="max"):
    if stat == "max":
        stats = zonal_stats(
            model,
            chm_path,
            stats=["max", "count"],
            nodata=-9999,
            all_touched=False,
        )
        model = model.copy()
        model["model_height_m"] = [s["max"] for s in stats]

    elif stat == "percentile_95":
        stats = zonal_stats(
            model,
            chm_path,
            stats=["percentile_95", "count"],
            nodata=-9999,
            all_touched=False,
        )
        model = model.copy()
        model["model_height_m"] = [
            s["percentile_95"] for s in stats
        ]

    else:
        raise ValueError("stat must be 'max' or 'percentile_95'")

    model["model_chm_pixel_count"] = [s["count"] for s in stats]
    return model


def get_matched_model_heights(field, model, iou_matrix, min_iou=0.3):
    """
    Match fieldwork crowns to model crowns and return a table of
    model-crown heights for matched fieldwork trees.
    """
    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    records = []

    for field_pos, model_pos in zip(row_idx, col_idx):
        iou = iou_matrix[field_pos, model_pos]

        if iou < min_iou:
            continue

        field_row = field.iloc[field_pos]
        model_row = model.iloc[model_pos]

        records.append({
            "plot": field_row["plot"],
            "tree_id": field_row["tree_id"],
            "model_height_m": model_row["model_height_m"],
            "model_chm_pixel_count": model_row["model_chm_pixel_count"],
            "iou": iou,
            "field_id": field_row["field_id"],
            "model_id": model_row["model_id"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Process plots
# ---------------------------------------------------------------------------

all_matched_heights = []

for plot in plot_names:
    print(f"Processing {plot}")

    pred_path = pred_dir / f"{plot}.gpkg"
    ref_path = ref_dir / plot / f"{plot}_crowns.gpkg"
    chm_path = chm_dir / plot / f"{plot}_CHM.tif"

    field, model = load_crowns(ref_path, pred_path)

    # Add plot name and unique IDs
    field["plot"] = plot
    field["field_id"] = [f"{plot}_field_{i}" for i in range(len(field))]
    model["model_id"] = [f"{plot}_model_{i}" for i in range(len(model))]

    # Extract model-crown CHM heights
    model = extract_model_chm_height(
        model,
        chm_path=chm_path,
        stat="max",  # change to "percentile_95" if preferred
    )

    # Match fieldwork crowns to model crowns
    iou_matrix = compute_iou_matrix(field, model)

    matched_heights = get_matched_model_heights(
        field=field,
        model=model,
        iou_matrix=iou_matrix,
        min_iou=MIN_IOU,
    )

    all_matched_heights.append(matched_heights)

# ---------------------------------------------------------------------------
# Combine and save CSV
# ---------------------------------------------------------------------------

matched_heights_all = pd.concat(
    all_matched_heights, ignore_index=True
)

print(matched_heights_all.head())
print(f"\nTotal matched crowns: {len(matched_heights_all)}")

matched_heights_all.to_csv(OUTPUT_CSV, index=False)