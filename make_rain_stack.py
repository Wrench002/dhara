import rasterio
import numpy as np

input_path = "data/dynamic/punjab_rainfall_7day_2019.tif"
output_path = "data/dynamic/rain_stack.tif"

TIME_STEPS = 24

with rasterio.open(input_path) as src:
    rain = src.read(1)
    meta = src.meta.copy()

# Stack 24 identical rainfall layers
stack = np.stack([rain] * TIME_STEPS, axis=0)

meta.update({
    "count": TIME_STEPS,
    "dtype": "float32"
})

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(stack.astype(np.float32))

print("rain_stack.tif created successfully.")