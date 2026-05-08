import numpy as np
import rasterio
with rasterio.open("data/static/dem.tif") as src:
    print(src.shape)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

STATIC_FILES = ["dem.tif", "slope.tif", "hand.tif", "dist_river.tif", "lulc.tif"]

def read_stack(folder, files):
    arrays = []
    for f in files:
        path = os.path.join(folder, f)
        print(f" -> Loading: {f}")
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            img[img < -9000] = 0 
            arrays.append(img)
    return np.stack(arrays, axis=0)

if __name__ == "__main__":
    print("--- COMPILING DATA CUBES ---")
    static_cube = read_stack(config.STATIC_DIR, STATIC_FILES)
    
    with rasterio.open(os.path.join(config.DYNAMIC_DIR, "rain_stack.tif")) as src:
        dynamic_cube = src.read().astype(np.float32)

    labels = read_stack(config.LABEL_DIR, ["flood_label.tif", "landslide_label.tif"])

    np.save(os.path.join(config.DATA_DIR, "static_cube.npy"), static_cube)
    np.save(os.path.join(config.DATA_DIR, "dynamic_cube.npy"), dynamic_cube)
    np.save(os.path.join(config.DATA_DIR, "labels_cube.npy"), labels)
    print("\n[OK] DATACUBES SAVED")