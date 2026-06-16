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