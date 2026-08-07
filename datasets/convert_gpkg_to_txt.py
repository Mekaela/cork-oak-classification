import geopandas as gpd
import os
from PIL import Image
from shapely.geometry import box



# --- CONFIGURATION ---
# Folder containing your .gpkg and .tif files
DATA_FOLDER = "datasets/images/train" 
OUTPUT_DIR = "datasets/labels"
LAYER_NAME = None                 # Set to None to auto-detect the layer (Recommended)
CLASS_COLUMN = "id"               # Since you only have Cork Oak, we'll use 'id' as a placeholder, but it won't matter as we force class 0
# If you want to be safe, you can just ignore the column and force class 0.
CLASS_ID = 0                      # Cork Oak is always class 0


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

def get_image_dims(base_id):
    """Finds a .tif file starting with base_id"""
    for f in os.listdir(DATA_FOLDER):
        if f.endswith('.tif') or f.endswith('.tiff'):
            if f.startswith(base_id):
                try:
                    with Image.open(os.path.join(DATA_FOLDER, f)) as img:
                        return img.size
                except:
                    continue
    return None

def process_gpkg(gpkg_file, img_w, img_h):
    """
    Reads a GPKG, converts polygons to YOLO format,
    and writes ONE .txt file per plot (not per tree).
    """
    # Extract Base ID: "plot10_crowns.gpkg" -> "plot10"
    base_id = gpkg_file.replace("_crowns.gpkg", "").replace(".gpkg", "")
    
    # Load Data
    path = os.path.join(DATA_FOLDER, gpkg_file)
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"❌ Error reading {gpkg_file}: {e}")
        return

    if gdf.empty:
        print(f"⚠️  No data in {gpkg_file}")
        return

    # Get Global Bounds for Scaling
    minx, miny, maxx, maxy = gdf.total_bounds
    real_w = maxx - minx
    real_h = maxy - miny

    if real_w <= 0 or real_h <= 0:
        print(f"❌ Invalid bounds in {gpkg_file}")
        return

    # Scaling Factors
    scale_x = img_w / real_w
    scale_y = img_h / real_h

    # List to hold all lines for this plot
    lines = []

    for idx, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue
        
        x_min, y_min, x_max, y_max = poly.bounds
        
        # Calculate Center (GIS)
        c_x_gis = (x_min + x_max) / 2
        c_y_gis = (y_min + y_max) / 2
        
        # Convert to Pixel (Flip Y axis for Image)
        c_x_px = (c_x_gis - minx) * scale_x
        c_y_px = (maxy - c_y_gis) * scale_y
        
        # Width/Height in Pixels
        w_px = (x_max - x_min) * scale_x
        h_px = (y_max - y_min) * scale_y
        
        # Normalize to 0-1
        c_x = c_x_px / img_w
        c_y = c_y_px / img_h
        w = w_px / img_w
        h = h_px / img_h
        
        # Format: class x y w h
        lines.append(f"{CLASS_ID} {c_x:.6f} {c_y:.6f} {w:.6f} {h:.6f}")

    # Write ONE file per plot
    label_filename = f"{base_id}.txt"
    label_path = os.path.join(OUTPUT_DIR, "labels", label_filename)
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Copy the image to the output folder
    # Find the matching image
    img_src = None
    for f in os.listdir(DATA_FOLDER):
        if f.startswith(base_id) and (f.endswith('.tif') or f.endswith('.tiff')):
            img_src = f
            break
    
    if img_src:
        img_dst = os.path.join(OUTPUT_DIR, "images", img_src)
        import shutil
        shutil.copy(os.path.join(DATA_FOLDER, img_src), img_dst)
        print(f"✅ Processed {gpkg_file} -> {label_filename} ({len(lines)} trees)")
    else:
        print(f"⚠️  Processed {gpkg_file} -> {label_filename}, but image not found.")

def main():
    ensure_dirs()
    
    gpkg_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.gpkg')]
    if not gpkg_files:
        print("No .gpkg files found.")
        return

    print(f"Processing {len(gpkg_files)} plots...\n")

    for f in gpkg_files:
        base_id = f.replace("_crowns.gpkg", "").replace(".gpkg", "")
        dims = get_image_dims(base_id)
        
        if dims is None:
            print(f"⚠️  No image found for {base_id}. Skipping.")
            continue
        
        img_w, img_h = dims
        process_gpkg(f, img_w, img_h)

    print(f"\n✅ Conversion Complete!")
    print(f"   Labels: {OUTPUT_DIR}/labels/")
    print(f"   Images: {OUTPUT_DIR}/images/")
    print(f"   Format: One .txt per image (e.g., plot10.txt)")

if __name__ == "__main__":
    main()

# file_path = DATA_FOLDER + "/plot10_crowns.gpkg"  # Update path

# try:
#     # Read the file
#     gdf = gpd.read_file(file_path)
    
#     print(f"File: {file_path}")
#     print(f"Total rows (features): {len(gdf)}")
#     print(f"Columns: {list(gdf.columns)}")
    
#     # Check if there is a geometry column
#     if 'geometry' in gdf.columns:
#         print(f"✅ Geometry column found!")
#         print(f"   Geometry type: {gdf['geometry'].geometry_type.unique()}")
#         print(f"   Is empty? {gdf['geometry'].is_empty.all()}")
        
#         # Try to get bounds
#         if not gdf['geometry'].is_empty.all():
#             bounds = gdf.total_bounds
#             print(f"   Bounds: {bounds}")
#         else:
#             print(f"   ⚠️  All geometries are empty (null).")
#     else:
#         print(f"❌ NO geometry column found in this layer.")
#         print(f"   This file might be a table-only export.")
        
# except Exception as e:
#     print(f"❌ Error: {e}")

# Check gpkg names
# folder = DATA_FOLDER # Change to your folder name
# gpkg_files = [f for f in os.listdir(folder) if f.endswith('.gpkg')]

# print(f"Checking {len(gpkg_files)} files:\n")

# for f in gpkg_files:
#     path = os.path.join(folder, f)
#     try:
#         # Get the list of layers
#         layers = gpd.list_layers(path)
        
#         # Convert to list to avoid ambiguity
#         if isinstance(layers, list):
#             layer_list = layers
#         else:
#             # If it's a DataFrame or Series, convert it to a list of strings
#             layer_list = list(layers)
            
#         if layer_list:
#             print(f"File: {f} -> Layers: {layer_list}")
#         else:
#             print(f"File: {f} -> No layers found!")
            
#     except Exception as e:
#         print(f"File: {f} -> Error: {e}")

# print(f"Checking {len(gpkg_files)} files:\n")

# for f in gpkg_files:
#     path = os.path.join(folder, f)
#     try:
#         # list_layers returns a list of layer names
#         layers = gpd.list_layers(path)
#         if layers:
#             print(f"File: {f} -> Layers: {layers}")
#         else:
#             print(f"File: {f} -> No layers found!")
#     except Exception as e:
#         print(f"File: {f} -> Error: {e}")