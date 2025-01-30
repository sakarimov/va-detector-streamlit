import folium
import numpy as np
import joblib
import rasterio
import streamlit as st

from streamlit_folium import st_folium
from folium.raster_layers import ImageOverlay
from rasterio.enums import Resampling

with st.sidebar:
    path = st.file_uploader("Upload tif file")

# Load the pre-trained model
model_path = "storage/models/hist_gradient_boosting_model.pkl"
model = joblib.load(model_path)

# Path to input TIFF file
input_tiff = path if path else "storage/geotiff/202411100010.tiff"

# downsample factor
scale_factor = 2

# Load the TIFF file
with rasterio.open(input_tiff) as src:
    new_width = src.width // scale_factor
    new_height = src.height // scale_factor

    downsampled_data = src.read(
        out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear
    )

    # Get new spatial bounds
    new_transform = src.transform * src.transform.scale(
        (src.width / new_width), (src.height / new_height)
    )

    bounds = src.bounds

    folium_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    bands = []
    for i in range(1, 5):  # Assuming 4 bands
        bands.append(src.read(i))

    # Stack bands to create a feature array
    stacked_bands = np.stack(downsampled_data[:4], axis=-1)

    # Reshape the array for prediction
    reshaped_data = stacked_bands.reshape(-1, stacked_bands.shape[-1])

    # Make predictions
    predictions = model.predict(reshaped_data)

    # Reshape predictions back to image shape
    prediction_image = predictions.reshape(stacked_bands.shape[:2])

# Normalize the raster data for visualization
min_val, max_val = np.min(prediction_image), np.max(prediction_image)
normalized_image = (prediction_image - min_val) / (max_val - min_val)  # Scale to [0,1]

# Convert to RGBA format (Red for white, Transparent for black)
rgba_array = np.zeros((*normalized_image.shape, 4), dtype=np.uint8)
rgba_array[..., 0] = (normalized_image * 255).astype(np.uint8)  # Red channel
rgba_array[..., 1] = 0  # No Green
rgba_array[..., 2] = 0  # No Blue
rgba_array[..., 3] = (normalized_image * 255).astype(np.uint8)  # Alpha transparency

print("Creating folium map...")
folium_map = folium.Map()
folium_map.fit_bounds(folium_bounds)

# Add raster layer to Folium
print("Adding raster layer to Folium map...")
ImageOverlay(
    rgba_array,
    bounds=folium_bounds,
    opacity=1,
).add_to(folium_map)

# Render the map as HTML
folium.TileLayer("OpenStreetMap").add_to(folium_map)
folium_map_html = folium_map._repr_html_()

st_data = st_folium(folium_map, width=725, height=325)

st.write(downsampled_data.shape)
