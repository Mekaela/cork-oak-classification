import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rasterstats import zonal_stats
from scipy.optimize import linear_sum_assignment
from shapely.validation import make_valid


variable_dict = {
    "ht_m": ["Fieldwork Height", "(m)"],
    "model_height_m": ["UAV Height", "(m)"],
    'area_m2': ["Reference Polygon Area", "(m^2)"],
    'model_area_m2': ["Polygon Area", "(m^2)"]
}

model_dict = {
    "yolo_site": "YOLOv8 Site",
    "yolo_plots": "YOLOv8 Plots",
    "watershed": "Watershed",
    "detectree2_plots": "Detectree2 Plots",
    "detectree2_site": "Detectree2 Site"
}

def compute_plot_metrics(pred_gdf: gpd.GeoDataFrame,
                         ref_gdf: gpd.GeoDataFrame,
                         iou_threshold: float = 0.5) -> dict:
    """
    Compute detection metrics for a single plot.
    Returns a dict with TP, FP, FN, precision, recall, f1, accuracy,
    mean_iou, n_pred, and n_ref.
    """
    # Ensure same CRS
    ref_gdf = ref_gdf.to_crs(pred_gdf.crs)

    # Precompute areas (optional but useful)
    pred_gdf = pred_gdf.copy()
    ref_gdf = ref_gdf.copy()
    pred_gdf["area"] = pred_gdf.geometry.area
    ref_gdf["area"] = ref_gdf.geometry.area

    # Compute IoU for all pairs
    ious = []
    for i, p in pred_gdf.iterrows():
        for j, r in ref_gdf.iterrows():
            inter = p.geometry.intersection(r.geometry).area
            union = p.geometry.union(r.geometry).area
            iou = inter / union if union > 0 else 0.0
            ious.append((i, j, iou))

    iou_df = pd.DataFrame(ious, columns=["pred_id", "ref_id", "iou"])

    # Greedy one-to-one matching
    iou_df = iou_df.sort_values("iou", ascending=False)
    matched_pred = set()
    matched_ref = set()
    tp = 0
    matched_ious = []

    for _, row in iou_df.iterrows():
        if row["iou"] < iou_threshold:
            continue
        if row["pred_id"] in matched_pred or row["ref_id"] in matched_ref:
            continue

        matched_pred.add(row["pred_id"])
        matched_ref.add(row["ref_id"])
        tp += 1
        matched_ious.append(row["iou"])

    n_pred = len(pred_gdf)
    n_ref = len(ref_gdf)

    fp = n_pred - len(matched_pred)
    fn = n_ref - len(matched_ref)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denominator = tp + fp + fn
    accuracy = tp / denominator if denominator > 0 else 0.0

    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "n_pred": n_pred,
        "n_ref": n_ref,
    }


def compute_overall_metrics(metrics_df):
    # Reorder columns if you like
    cols = ["plot_id", "TP", "FP", "FN", "precision", "recall", "accuracy", "f1", "mean_iou", "n_pred", "n_ref"]
    metrics_df = metrics_df[cols]

    TP_total = metrics_df["TP"].sum()
    FP_total = metrics_df["FP"].sum()
    FN_total = metrics_df["FN"].sum()
    total_detections = TP_total + FP_total

    precision_overall = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    recall_overall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0
    f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall) if (precision_overall + recall_overall) > 0 else 0.0
    accuracy_overall = (
        TP_total / (TP_total + FP_total + FN_total)
        if (TP_total + FP_total + FN_total) > 0
        else 0.0
    )

    mean_iou_overall = metrics_df["mean_iou"].mean()  # average of per-plot mean IoU

    print("Overall metrics across all plots:")
    print(f"TP: {TP_total}, FP: {FP_total}, FN: {FN_total}")
    print(f"Precision: {precision_overall:.3f}")
    print(f"Recall: {recall_overall:.3f}")
    print(f"Accuracy: {accuracy_overall:.3f}")
    print(f"F1: {f1_overall:.3f}")
    print(f"Mean IoU (avg of plot means): {mean_iou_overall:.3f}")
    print(f"Total detections: {total_detections}")


def merge_filter_dfs(preditions_df, reference, crown_areas):
    reference['tree_id'] = ('p' + reference['plot_id'].astype(str).str.strip() + 't' + reference['tree_id'].astype(str).str.strip())
    merged_heights = pd.merge(reference, preditions_df, on='tree_id', how='inner')
    merged_heights_df = merged_heights[["tree_id", "ht_m", "model_height_m", "model_chm_pixel_count"]].copy()
    filtered_df = merged_heights_df[(merged_heights_df['model_height_m'] != 0) & (merged_heights_df['model_height_m'].notna())]
    merged_areas = pd.merge(filtered_df, crown_areas, on='tree_id', how='inner')
    merged_areas["model_area_m2"] = merged_areas["model_chm_pixel_count"]/100
    return merged_areas

def scatterplot(df, model, x, y):
    # add linear regression line
    r2 = r2_score(df[x], df[y])
    rmse = np.sqrt(mean_squared_error(df[x], df[y]))
    n = len(df)
    coeffs = np.polyfit(df[x], df[y], 1)
    trend_x = np.linspace(df[x].min(), df[x].max(), 100)
    trend_y = np.polyval(coeffs, trend_x)

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], alpha=0.7)
    plt.plot(trend_x, trend_y, color='red', linewidth=2)
    plt.text(0.05, 0.95, f'$R^2$={r2:.3f}  RMSE={rmse:.2f}  n={n}',
             transform=plt.gca().transAxes, va='top', fontsize=10)
    plt.legend()
    plt.xlabel(f'{model_dict[model]} {variable_dict[x][0]} {variable_dict[x][1]}')
    plt.ylabel(f'{variable_dict[y][0]} {variable_dict[y][1]}')
    # plt.title(f'{model_dict[model]} {variable_dict[x][0]} vs {variable_dict[y][0]}')
    # plt.title(f'{model_dict[model]}')
    plt.grid(True)
    plt.show()

# Assuming your DataFrame is named 'df' with columns: 'tree_id', 'ht_m', 'model_height_m'
def calculate_metrics(df, type, x, y):
    # 1. Calculate Metrics
    r2 = r2_score(df[x], df[y])
    rmse = np.sqrt(mean_squared_error(df[x], df[y]))
    bias = (df[y] - df[x]).mean()
    mae = mean_absolute_error(df[x], df[y])
    pearson_r, _ = stats.pearsonr(df[x], df[y])
    spearman_r, _ = stats.spearmanr(df[x], df[y])

    # 2. Create a dictionary of results
    metrics_dict = {
        "Metric": ["R²", "RMSE", "Bias", "MAE", "Pearson R", "Spearman R"],
        "Value": [r2, rmse, bias, mae, pearson_r, spearman_r]
    }

    # 3. Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # 4. Save to CSV
    # metrics_df.to_csv(f"{model}_{type}_metrics.csv", index=False)
    print(metrics_df)
    return metrics_df

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


def match_all_plots(plot_names, pred_dir, ref_dir, chm_dir, min_iou):
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
            min_iou=min_iou,
        )

        all_matched_heights.append(matched_heights)
    matched_heights_all = pd.concat(
        all_matched_heights, ignore_index=True
    )

    return matched_heights_all
