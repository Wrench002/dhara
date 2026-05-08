import rasterio
import numpy as np

with rasterio.open("data/static/dem.tif") as src:
    meta = src.meta.copy()
    zeros = np.zeros((src.height, src.width), dtype=np.uint8)

# Update metadata safely
meta.update({
    "count": 1,
    "dtype": "uint8",
    "nodata": 0  # valid for uint8
})

with rasterio.open("data/labels/landslide_label.tif", "w", **meta) as dst:
    dst.write(zeros, 1)

print("landslide_label.tif created successfully.")