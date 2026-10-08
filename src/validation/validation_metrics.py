import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import date


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

def save_metrics(metrics_df, model_name):
    # Reorder columns if you like
    cols = ["plot_id", "TP", "FP", "FN", "precision", "recall", "accuracy", "f1", "mean_iou", "n_pred", "n_ref"]
    metrics_df = metrics_df[cols]

    print(metrics_df.to_string(index=False))
    # Optionally save
    today = date.today()
    metrics_df.to_csv(f"plot_metrics_{model_name}_{today}.csv", index=False)

    TP_total = metrics_df["TP"].sum()
    FP_total = metrics_df["FP"].sum()
    FN_total = metrics_df["FN"].sum()
    total_detections = TP_total + FP_total + FN_total

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

    # Overall metrics as a one-row DataFrame
    overall_row = pd.DataFrame([{
        "TP": TP_total,
        "FP": FP_total,
        "FN": FN_total,
        "precision": precision_overall,
        "recall": recall_overall,
        "f1": f1_overall,
        "mean_iou": mean_iou_overall,
    }])
    overall_row.to_csv(f"plot_metrics_overall_{model_name}_{today}.csv", index=False)

# ------------------------------------------------------------------
# Iterate over test plots
# ------------------------------------------------------------------

model = "yolo_plots"
# change this to the directory for the predicted model
pred_dir = Path(f"src/validation/{model}")
ref_dir = Path("data/plots")

# for only the test plots:
plot_names = ["plot61","plot175","plot238"]  

results = []

for plot in plot_names:
    pred_path = pred_dir / f"{plot}.gpkg"
    ref_path = ref_dir / f"{plot}" / f"{plot}_crowns.gpkg"

    print(pred_path, ref_path)

    if not pred_path.exists() or not ref_path.exists():
        print(f"Skipping {plot}: missing files")
        continue

    pred_gdf = gpd.read_file(pred_path)
    ref_gdf = gpd.read_file(ref_path)

    metrics = compute_plot_metrics(pred_gdf, ref_gdf, iou_threshold=0.5)
    metrics["plot_id"] = plot
    results.append(metrics)

metrics_df = pd.DataFrame(results)

save_metrics(metrics_df, model)

