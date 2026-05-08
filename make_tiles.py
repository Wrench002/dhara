import numpy as np
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

static_cube = np.load(os.path.join(DATA_DIR, "static_cube.npy"))
dynamic_cube = np.load(os.path.join(DATA_DIR, "dynamic_cube.npy"))
labels_cube = np.load(os.path.join(DATA_DIR, "labels_cube.npy"))

Cs, H, W = static_cube.shape
tile_id = 0

for i in range(0, H - TILE_SIZE + 1, TILE_SIZE):
    for j in range(0, W - TILE_SIZE + 1, TILE_SIZE):
        x_s = static_cube[:, i:i+TILE_SIZE, j:j+TILE_SIZE]
        x_d = dynamic_cube[-TIME_STEPS:, i:i+TILE_SIZE, j:j+TILE_SIZE]
        y = labels_cube[:, i:i+TILE_SIZE, j:j+TILE_SIZE]

        if np.isnan(y).any() or y.sum() < 10:
            continue

        save_dir = TRAIN_TILES_DIR if random.random() < 0.85 else VAL_TILES_DIR
        np.savez_compressed(os.path.join(save_dir, f"tile_{tile_id:05d}.npz"), x_static=x_s, x_dynamic=x_d, y=y)
        tile_id += 1

print(f"[OK] Tiles Generated: {tile_id}")