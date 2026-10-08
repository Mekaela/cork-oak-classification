from pathlib import Path
from datetime import date
import pandas as pd
import geopandas as gpd
from classification_validation import print_total_polygon_area, evaluate_all_plots, calculate_pooled_accuracy
from detection_validation import match_all_plots, compute_plot_metrics, compute_overall_metrics, merge_filter_dfs, calculate_metrics, scatterplot

model = "yolo_plots"
today = date.today()
total = 0
pred_dir = Path(f"src/validation/{model}")
ref_dir = Path("data/plots")
chm_dir = Path("data/plots")

if __name__ == "__main__":
    # preds = pd.read_csv(f"2026-10-07_{model}_matched_model_crown_heights.csv")
    ref = pd.read_csv('notebooks/lizirias_fieldwork_trees.csv')
    crown_areas = pd.read_csv("notebooks/crown_areas_trees.csv")

    plot_names = ["plot61", "plot175", "plot238"]
    min_iou = 0.5

    # match crowns
    matched_df = match_all_plots(plot_names, pred_dir, ref_dir, chm_dir, min_iou)

    # Height validation
    merged_df = merge_filter_dfs(matched_df, ref, crown_areas)
    print("Height metrics:")
    x = 'model_height_m'
    y = 'ht_m'
    # Figure 6 in report
    # scatterplot(merged_df, model, x, y)
    # Table 3 in the report
    calculate_metrics(merged_df, 'height', x, y)

    # Area validation
    print("Area metrics:")
    x = 'area_m2'
    y = 'model_area_m2'
    # Figure 7 in report
    # scatterplot(merged_df, model, x, y)
    # Table 4 in the report
    calculate_metrics(merged_df, 'crown area', x, y)

    # polygon area and detection metrics
    results = []
    for plot in plot_names:
        # for reference plots:
        # gpkg_path = f"data/plots/{plot}/{plot}_crowns.gpkg"
        # for model predictions: 
        pred_path = pred_dir / f"{plot}.gpkg"
        ref_path = ref_dir / f"{plot}" / f"{plot}_crowns.gpkg"

        total_area = print_total_polygon_area(pred_path, plot)
        total += total_area

        if not pred_path.exists() or not ref_path.exists():
                    print(f"Skipping {plot}: missing files")
                    continue

        pred_gdf = gpd.read_file(pred_path)
        ref_gdf = gpd.read_file(ref_path)

        metrics = compute_plot_metrics(pred_gdf, ref_gdf, min_iou)
        metrics["plot_id"] = plot
        results.append(metrics)

    # Table 5 in report
    print("Total area: ", total)
    # Table 2 in report
    metrics_df = pd.DataFrame(results)
    print(metrics_df)

    # need overall values of above detection - Table 1 in report
    compute_overall_metrics(metrics_df)

    # classification validation
    results_df = evaluate_all_plots(
        plot_names,
        pred_dir, 
        ref_dir
    )

    calculate_pooled_accuracy(results_df)

