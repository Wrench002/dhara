import os
import numpy as np
import rasterio

static_folder = "data/static"
dynamic_path = "data/dynamic/rain_stack.tif"
label_folder = "data/labels"
output_folder = "tiles"

os.makedirs(output_folder, exist_ok=True)

TILE_SIZE = 256

# ---------------------------
# Load static layers
# ---------------------------
static_files = [
    "dem.tif",
    "slope.tif",
    "hand.tif",
    "dist_river.tif",
    "lulc.tif"
]

static_arrays = []
for f in static_files:
    with rasterio.open(os.path.join(static_folder, f)) as src:
        arr = src.read(1).astype(np.float32)

        # Replace nodata & invalid values
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        static_arrays.append(arr)

x_static_full = np.stack(static_arrays, axis=0)

# ---------------------------
# Load dynamic
# ---------------------------
with rasterio.open(dynamic_path) as src:
    x_dynamic_full = src.read().astype(np.float32)

x_dynamic_full = np.nan_to_num(x_dynamic_full, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------
# Load labels
# ---------------------------
with rasterio.open(os.path.join(label_folder, "flood_label.tif")) as src:
    flood = src.read(1)

with rasterio.open(os.path.join(label_folder, "landslide_label.tif")) as src:
    landslide = src.read(1)

flood = (flood > 0.5).astype(np.float32)
landslide = (landslide > 0.5).astype(np.float32)

y_full = np.stack([flood, landslide], axis=0)

# ---------------------------
# Normalize safely
# ---------------------------

# Static
x_static_full[0] /= 1000.0
x_static_full[1] /= 90.0
x_static_full[2] /= 100.0
x_static_full[3] /= 1000.0
x_static_full[4] /= 100.0

# Dynamic
x_dynamic_full /= 200.0

# Final safety clamp
x_static_full = np.nan_to_num(x_static_full, nan=0.0, posinf=0.0, neginf=0.0)
x_dynamic_full = np.nan_to_num(x_dynamic_full, nan=0.0, posinf=0.0, neginf=0.0)

H, W = flood.shape
tile_id = 0

for i in range(0, H - TILE_SIZE, TILE_SIZE):
    for j in range(0, W - TILE_SIZE, TILE_SIZE):

        x_static = x_static_full[:, i:i+TILE_SIZE, j:j+TILE_SIZE]
        x_dynamic = x_dynamic_full[:, i:i+TILE_SIZE, j:j+TILE_SIZE]
        y = y_full[:, i:i+TILE_SIZE, j:j+TILE_SIZE]


        np.savez_compressed(
            os.path.join(output_folder, f"tile_{tile_id}.npz"),
            x_static=x_static,
            x_dynamic=x_dynamic,
            y=y
        )

        tile_id += 1

print(f"{tile_id} tiles created successfully.")