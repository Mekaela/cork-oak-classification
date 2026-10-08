from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np

def show_labeled_tile(ndvi_tile, labels_tile, vmin=-1, vmax=1):
    # ndvi_tile: (H, W) or (H, W, 1)
    if ndvi_tile.ndim == 3:
        ndvi_tile = ndvi_tile[..., 0]

    # Turn labels into RGB overlay
    overlay = label2rgb(labels_tile, image=ndvi_tile,
                        bg_label=0, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ndvi_tile, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0].set_title("NDVI")
    axes[0].axis("off")

    axes[1].imshow(labels_tile, cmap="nipy_spectral")
    axes[1].set_title("Labels")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# import cv2
# import os

# images_dir = "models/training/images/train"
# labels_dir = "models/training/labels"
# out_dir = "models/training/visuals"
# os.makedirs(out_dir, exist_ok=True)

# for label_file in os.listdir(labels_dir):
#     if not label_file.endswith(".txt"):
#         continue
#     base = os.path.splitext(label_file)[0]
#     img_path = os.path.join(images_dir, f"{base}.tif")  # adjust extension
#     if not os.path.exists(img_path):
#         continue

#     img = cv2.imread(img_path)
#     h, w = img.shape[:2]

#     with open(os.path.join(labels_dir, label_file)) as f:
#         for line in f:
#             parts = list(map(float, line.strip().split()))
#             cls = int(parts[0])
#             xc, yc, bw, bh = parts[1:5]

#             x_center = xc * w
#             y_center = yc * h
#             width = bw * w
#             height = bh * h

#             x1 = int(x_center - width/2)
#             y1 = int(y_center - height/2)
#             x2 = int(x_center + width/2)
#             y2 = int(y_center + height/2)

#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img, str(cls), (x1, y1-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     cv2.imwrite(os.path.join(out_dir, f"{base}_vis.jpg"), img)


import os
import glob
import cv2

# ---- CONFIG ----
images_dir = "datasets/images/train"
labels_dir = "datasets/labels/train"
vis_out_dir = "datasets/visuals"   
os.makedirs(vis_out_dir, exist_ok=True)

import numpy as np
import cv2

def draw_yolo_polygons(img, label_path):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img  # no labels, just return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        if len(parts) < 3:
            continue

        class_id = int(parts[0])
        coords = parts[1:]

        # Must be even number (x, y pairs)
        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            nx = coords[i]
            ny = coords[i+1]
            x = nx * w
            y = ny * h
            pts.append([x, y])

        # Convert to NumPy array with correct shape and type
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(img, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)

        # Optional: label with class id at first vertex
        if len(pts) > 0:
            x0, y0 = int(round(pts[0][0])), int(round(pts[0][1]))
            cv2.putText(
                img, str(class_id), (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

    return img

# Build a map from base name -> label file
label_map = {}
for lbl in glob.glob(os.path.join(labels_dir, "*.txt")):
    base = os.path.splitext(os.path.basename(lbl))[0]
    label_map[base] = lbl


for img_path in glob.glob(os.path.join(images_dir, f"*.tif")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = label_map.get(base)

    img = cv2.imread(img_path)
    if img is None:
        continue

    if label_path:
        img = draw_yolo_polygons(img, label_path)

    out_path = os.path.join(vis_out_dir, f"{base}_vis.tif")
    cv2.imwrite(out_path, img)

print("Done. Visualised images written to:", vis_out_dir)