import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

dem_path = "data/static/dem.tif"
rain_input = "data/dynamic/rain_stack.tif"
rain_output = "data/dynamic/rain_stack_fixed.tif"

with rasterio.open(dem_path) as dem:
    dem_meta = dem.meta.copy()
    dem_transform = dem.transform
    dem_crs = dem.crs
    dem_shape = (dem.height, dem.width)

with rasterio.open(rain_input) as src:
    rain_data = src.read()  # (time, H, W)
    rain_transform = src.transform
    rain_crs = src.crs

T = rain_data.shape[0]

# Create empty aligned array
rain_resampled = np.zeros((T, dem_shape[0], dem_shape[1]), dtype=np.float32)

for t in range(T):
    reproject(
        source=rain_data[t],
        destination=rain_resampled[t],
        src_transform=rain_transform,
        src_crs=rain_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear
    )

dem_meta.update({
    "count": T,
    "dtype": "float32",
    "nodata": None
})

with rasterio.open(rain_output, "w", **dem_meta) as dst:
    dst.write(rain_resampled)

print("rain_stack_fixed.tif created successfully.")