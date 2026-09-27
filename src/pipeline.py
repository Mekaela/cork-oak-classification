import rasterio
from scipy.ndimage import grey_erosion, binary_erosion, convolve
import numpy as np
# Main pipeline: load → segment → extract → predict

def estimate_dtm_from_dsm(
    dsm,
    valid_mask,
    window_size=201
):
    if dsm.ndim != 3 or dsm.shape[2] != 1:
        raise ValueError("Expected dsm shape (H, W, 1)")

    if valid_mask.shape != dsm.shape[:2]:
        raise ValueError("valid_mask must have shape (H, W)")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    dsm_float = dsm.astype(np.float32, copy=False)

    valid = (
        valid_mask
        & np.isfinite(dsm_float[..., 0])
    )

    # Invalid pixels cannot become the local minimum.
    filtered = np.where(
        valid[..., np.newaxis],
        dsm_float,
        np.inf
    )

    dtm = grey_erosion(
        filtered,
        size=(window_size, window_size, 1),
        mode="constant",
        cval=np.inf,
        origin=0
    )

    dtm[..., 0][~valid] = np.nan

    return dtm

def compute_chm_from_dsm(dsm, dtm, mask, nodata=0.0):
    chm = dsm - dtm                      # (H, W, 1)

    # Apply polygon mask: keep only inside, set outside to nodata
    chm_out = chm.copy()
    chm_out[~mask, 0] = nodata          # mask is (H, W), chm_out is (H, W, 1)

    edge_margin = 100   # smaller amounts were not enough
    # inner_mask = binary_erosion(mask, iterations=edge_margin)
    inner_mask = binary_erosion(mask)
    chm_display = chm_out.copy()
    chm_display[~inner_mask, 0] = np.nan  # hide outer band

    return chm_display

def save_layer(chm, profile, out_path, nodata=0.0):
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

def build_chm_mask(chm):
    return (chm[:, :, 0] > 2.0).astype(np.uint8)


# to visualise creating the CHM
# plt.imshow(chm[:, :, 0], cmap="terrain")
# plt.colorbar()
# plt.axis("off")
# plt.show()

# to save the CHM:
# save_chm(chm, profile, "20250728-ISA-CHM-10cm.tif", nodata=0.0)
