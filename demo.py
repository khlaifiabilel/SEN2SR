# rm -rf dist/ build/ .egg-info; poetry build; python3 -m twine upload dist/*

import mlstac
import torch
import cubo

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/Reference_RSWIR_x2/mlm.json",
  output_dir="model/SEN2SRLite_Reference_RSWIR_x2",
)

mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/Reference_RSWIR_x2/mlm.json",
  output_dir="model/SEN2SR_Reference_RSWIR_x2",
)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)

# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to("cpu")

# Load the model
device = "cpu"
model1 = mlstac.load("model/SEN2SR_Reference_RSWIR_x2").compiled_model(device=device)
model1 = model1.to(device)
model2 = mlstac.load("model/SEN2SRLite_Reference_RSWIR_x2").compiled_model(device=device)
model2 = model2.to(device)

# Apply model
superX1 = model1(X[None]).squeeze(0)
superX2 = model2(X[None]).squeeze(0)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.moveaxis(original_s2_numpy[[9, 7, 5], 32:64, 32:64], 0, -1)*1.4)
ax[0].set_title("S2 B12/B8/B7 - 20m")
ax[1].imshow(np.moveaxis(superX1[[9, 7, 5], 32:64, 32:64].cpu().numpy(), 0, -1)*1.4)
ax[1].set_title("SEN2SRLite B12/B8/B7 - 10m")
ax[2].imshow(np.moveaxis(superX2[[9, 7, 5], 32:64, 32:64].cpu().numpy(), 0, -1)*1.4)
ax[2].set_title("SEN2SR B12/B8/B7 - 10m")
plt.show()