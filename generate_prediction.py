import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import rasterio
from model_lstm_multitask import MultiTaskLSTM

device = torch.device("cpu")

# Load model
model = MultiTaskLSTM()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load full static
static_folder = "data/static"
dynamic_path = "data/dynamic/rain_stack.tif"

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
        static_arrays.append(src.read(1))

x_static = np.stack(static_arrays, axis=0).astype(np.float32)

with rasterio.open(dynamic_path) as src:
    x_dynamic = src.read().astype(np.float32)
    meta = src.meta

# SAME NORMALIZATION AS TRAINING
x_static[0] /= 1000.0
x_static[1] /= 90.0
x_static[2] /= 100.0
x_static[3] /= 1000.0
x_static[4] /= 100.0
x_dynamic /= 200.0

x_static = np.nan_to_num(x_static)
x_dynamic = np.nan_to_num(x_dynamic)

# Convert to tensors
x_s = torch.tensor(x_static).unsqueeze(0)
x_d = torch.tensor(x_dynamic).unsqueeze(0)

with torch.no_grad():
    f_logits, _ = model(x_s, x_d)
    flood_prob = torch.sigmoid(f_logits).squeeze().numpy()

# Save GeoTIFF
meta.update({
    "count": 1,
    "dtype": "float32"
})

with rasterio.open("outputs/flood_prediction.tif", "w", **meta) as dst:
    dst.write(flood_prob, 1)

print("Flood prediction saved to outputs/flood_prediction.tif")