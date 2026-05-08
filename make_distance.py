import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

# Paths
dem_path = "data/static/dem.tif"
river_path = "data/static/punjab_rivers.shp"
output_path = "data/static/dist_river.tif"

# Load DEM (reference grid)
with rasterio.open(dem_path) as dem:
    meta = dem.meta.copy()
    transform = dem.transform
    shape = (dem.height, dem.width)

# Load rivers
rivers = gpd.read_file(river_path)

# Rasterize rivers
river_raster = rasterize(
    [(geom, 1) for geom in rivers.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

# Compute distance transform
distance = distance_transform_edt(river_raster == 0)

# Convert pixel distance to degrees (same resolution as DEM)
pixel_size = abs(transform.a)
distance = distance * pixel_size

# Save raster
meta.update({
    "count": 1,
    "dtype": "float32"
})

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(distance.astype(np.float32), 1)

print("dist_river.tif generated successfully.")