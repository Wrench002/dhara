import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

dem_path = "data/static/dem.tif"
flood_input = "data/labels/flood_label.tif"
flood_output = "data/labels/flood_label_fixed.tif"

with rasterio.open(dem_path) as dem:
    dem_meta = dem.meta.copy()
    dem_transform = dem.transform
    dem_crs = dem.crs
    dem_shape = (dem.height, dem.width)

with rasterio.open(flood_input) as src:
    flood_data = src.read(1)
    flood_transform = src.transform
    flood_crs = src.crs

# Create empty array matching DEM
flood_resampled = np.zeros(dem_shape, dtype=np.uint8)

reproject(
    source=flood_data,
    destination=flood_resampled,
    src_transform=flood_transform,
    src_crs=flood_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.nearest
)

dem_meta.update({
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})

with rasterio.open(flood_output, "w", **dem_meta) as dst:
    dst.write(flood_resampled, 1)

print("flood_label_fixed.tif created successfully.")