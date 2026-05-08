import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

dem_path = "data/static/dem.tif"
lulc_input = "data/static/lulc.tif"
lulc_output = "data/static/lulc_fixed.tif"

with rasterio.open(dem_path) as dem:
    dem_meta = dem.meta.copy()
    dem_transform = dem.transform
    dem_crs = dem.crs
    dem_shape = (dem.height, dem.width)

with rasterio.open(lulc_input) as src:
    lulc_data = src.read(1)
    lulc_transform = src.transform
    lulc_crs = src.crs

# Create empty target array
lulc_resampled = np.zeros(dem_shape, dtype=np.uint8)

reproject(
    source=lulc_data,
    destination=lulc_resampled,
    src_transform=lulc_transform,
    src_crs=lulc_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.nearest
)

# Clean metadata for uint8
dem_meta.update({
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})

with rasterio.open(lulc_output, "w", **dem_meta) as dst:
    dst.write(lulc_resampled, 1)

print("lulc_fixed.tif created successfully.")