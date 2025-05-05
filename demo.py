# rm -rf dist/ build/ .egg-info; poetry build; python3 -m twine upload dist/*
import mlstac
import sen2sr
import torch
import cubo

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=1024,
    resolution=10
)

# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Load the model
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)


# Apply model
superX = sen2sr.predict_large(
    model=model,
    X=X, # The input tensor
    overlap=16, # The overlap between the patches
)


import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(np.moveaxis(original_s2_numpy[[2, 1, 0]], 0, -1)*3)
ax[0].set_title("Original Image")
ax[1].imshow(np.moveaxis(superX[[2, 1, 0]].cpu().numpy(), 0, -1)*3)
ax[1].set_title("Super Resolved SEN2SRLite Image")
plt.show()