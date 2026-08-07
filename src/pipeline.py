import rasterio
from scipy.ndimage import grey_erosion
import numpy as np
from scipy.ndimage import binary_erosion
# Main pipeline: load → segment → extract → predict

def estimate_dtm_from_dsm(dsm, window_size=200):
    """
    dsm: (H, W, 1) array
    window_size: size of square erosion window in pixels (e.g. ~ crown diameter)
    """
    if dsm.ndim != 3 or dsm.shape[2] != 1:
        raise ValueError("Expected dsm shape (H, W, 1)")

    size = (window_size, window_size, 1)  # erode spatial dims only
    dtm = grey_erosion(dsm, size=size, mode='nearest') 
    return dtm

def compute_chm_from_dsm(dsm, mask, window_size=200, nodata=0.0):
    dtm = estimate_dtm_from_dsm(dsm, window_size=window_size)
    chm = dsm - dtm                      # (H, W, 1)

    # Apply polygon mask: keep only inside, set outside to nodata
    chm_out = chm.copy()
    chm_out[~mask, 0] = nodata          # mask is (H, W), chm_out is (H, W, 1)

    edge_margin = 100   # smaller amounts were not enough
    inner_mask = binary_erosion(mask, iterations=edge_margin)
    chm_display = chm_out.copy()
    chm_display[~inner_mask, 0] = np.nan  # hide outer band

    return chm_display

def save_chm(chm, profile, out_path, nodata=0.0):
    # chm: (H, W, 1)
    out = chm[:, :, 0].astype(np.float32)    # single band

    prof = profile.copy()
    prof.update(
        count=1,
        dtype="float32",
        nodata=nodata
    )

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)


    # to visualise creating the CHM
    # plt.imshow(chm[:, :, 0], cmap="terrain")
    # plt.colorbar()
    # plt.axis("off")
    # plt.show()

    # to save the CHM:
    save_chm(chm, profile, "20250728-ISA-CHM-10cm.tif", nodata=0.0)