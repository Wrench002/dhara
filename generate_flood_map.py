import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
import rasterio
from model_lstm_multitask import MultiTaskLSTM

device = torch.device("cpu")
TILE_SIZE = 256

start_time = time.time()

print("Loading model...")

# -----------------------------
# Load trained model
# -----------------------------
model = MultiTaskLSTM()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model loaded.")

# -----------------------------
# Load data
# -----------------------------
static_folder = "data/static"
dynamic_path = "data/dynamic/rain_stack.tif"

static_files = [
    "dem.tif",
    "slope.tif",
    "hand.tif",
    "dist_river.tif",
    "lulc.tif"
]

print("Loading static layers...")

static_arrays = []
for f in static_files:
    with rasterio.open(os.path.join(static_folder, f)) as src:
        static_arrays.append(src.read(1))

x_static = np.stack(static_arrays, axis=0).astype(np.float32)

with rasterio.open(dynamic_path) as src:
    x_dynamic = src.read().astype(np.float32)
    meta = src.meta.copy()

print("Applying normalization...")

# SAME normalization used during training
x_static[0] /= 1000.0
x_static[1] /= 90.0
x_static[2] /= 100.0
x_static[3] /= 1000.0
x_static[4] /= 100.0
x_dynamic /= 200.0

x_static = np.nan_to_num(x_static)
x_dynamic = np.nan_to_num(x_dynamic)

H, W = x_static.shape[1], x_static.shape[2]

prediction = np.zeros((H, W), dtype=np.float32)

tiles_x = (H - TILE_SIZE) // TILE_SIZE
tiles_y = (W - TILE_SIZE) // TILE_SIZE
total_tiles = tiles_x * tiles_y
tile_count = 0

print(f"Running tile-based inference on {total_tiles} tiles...")

# -----------------------------
# Tile inference
# -----------------------------
with torch.no_grad():
    for i in range(0, H - TILE_SIZE, TILE_SIZE):
        for j in range(0, W - TILE_SIZE, TILE_SIZE):

            tile_count += 1

            if tile_count % 50 == 0:
                print(f"Processed {tile_count}/{total_tiles} tiles")

            x_s_tile = x_static[:, i:i+TILE_SIZE, j:j+TILE_SIZE]
            x_d_tile = x_dynamic[:, i:i+TILE_SIZE, j:j+TILE_SIZE]

            x_s_tensor = torch.tensor(x_s_tile).unsqueeze(0)
            x_d_tensor = torch.tensor(x_d_tile).unsqueeze(0)

            f_logits, _ = model(x_s_tensor, x_d_tensor)
            flood_prob = torch.sigmoid(f_logits).squeeze().numpy()

            prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] = flood_prob

print("Inference complete.")

# -----------------------------
# Save outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

meta.update({
    "count": 1,
    "dtype": "float32"
})

prob_path = "outputs/flood_probability.tif"
binary_path = "outputs/flood_binary.tif"

with rasterio.open(prob_path, "w", **meta) as dst:
    dst.write(prediction, 1)

binary_map = (prediction > 0.5).astype(np.float32)

with rasterio.open(binary_path, "w", **meta) as dst:
    dst.write(binary_map, 1)

print("Saved:")
print(prob_path)
print(binary_path)

print(f"Total time: {round(time.time() - start_time, 2)} seconds")