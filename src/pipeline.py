from scipy.ndimage import grey_erosion
import numpy as np
# Main pipeline: load → segment → extract → predict

# def estimate_chm_from_dsm_tile(dsm_data):   
#     # 20m window size (bigger than largest crown - hopefully)
#     window_size = 50 
#     dtm_estimate = grey_erosion(dsm_data, size=(1, window_size, window_size, 1))
#     chm_estimate = dsm_data - dtm_estimate
#     return chm_estimate 

def estimate_chm_from_dsm_tile(dsm_data):   
    window_size = 200 
    
    # Get the number of dimensions in the tile
    ndim = dsm_data.ndim
    
    if ndim == 3:
        # Shape is likely (H, W, C). Erode only spatial dims (H, W).
        # Assuming format: (Height, Width, Channels)
        size = (window_size, window_size, 1) 
    elif ndim == 4:
        # Shape is likely (B, H, W, C) or similar
        size = (1, window_size, window_size, 1)
    else:
        raise ValueError(f"Unexpected input dimensions: {ndim}")

    dtm_estimate = grey_erosion(dsm_data, size=size)
    chm_estimate = dsm_data - dtm_estimate
    return chm_estimate

# estinate CHM from DSE:
# def estimate_chm_from_dsm(dsm_tiles):
#     chm_tiles = []
#     for tile in dsm_tiles:
#         chm_tile = estimate_chm_from_dsm_tile(tile)
#         chm_tiles.append(chm_tile)
#     return np.array(chm_tiles)


def estimate_chm_from_dsm(dsm_tiles):
    n_rows, n_cols = dsm_tiles.shape[0], dsm_tiles.shape[1]
    chm_list = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            tile = dsm_tiles[i, j]
            chm_tile = estimate_chm_from_dsm_tile(tile)
            chm_list.append(chm_tile)
    
    # Convert to array
    chm_array = np.array(chm_list)
    
    print(f"chm_array.shape: {chm_array.shape}")
    print(f"Expected final_shape: {(n_rows, n_cols) + chm_array.shape[1:]}")

    # Ensure all tiles have the same shape
    assert len(chm_array.shape) > 1, "chm_array should have more than 1 dimension"

    final_shape = (n_rows, n_cols) + chm_array.shape[1:]
    return chm_array.reshape(final_shape)
