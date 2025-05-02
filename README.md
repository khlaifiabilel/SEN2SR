# SEN2SR

<p align="center">
  <img src="assets/sen2sr.gif" width="90%">
</p>


<p align="center">
   <em>A Python package for enhancing the spatial resolution of Sentinel-2 satellite images up to 2.5 meters</em> 🚀
</p>


<p align="center">
<a href='https://pypi.python.org/pypi/sen2sr'>
    <img src='https://img.shields.io/pypi/v/sen2sr.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
<a href="https://colab.research.google.com/drive/1TD014aY145q1reKN644egUtIM6tIx9vH?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>


---

**GitHub**: [https://github.com/ESAOpenSR/sen2sr](https://github.com/ESAOpenSR/sen2sr) 🌐

**PyPI**: [https://pypi.org/project/sen2sr/](https://pypi.org/project/sen2sr/) 🛠️

---

## **Table of Contents**

- [**Overview**](#overview-)
- [**Installation** ⚙️](#installation-)
- [**Predict 10m and 20m bands**](#predict-10m-and-20m-bands)
- [**Predict only RGBNIR bands**](#predict-only-rgbnir-bands)
- [**Predict on large images**](#predict-on-large-images)
- [**Estimate the Local Attention Map of the model**](#estimate-the-local-attention-map-of-the-model-)


## **Overview**

**sen2sr** is a Python package designed to enhance the spatial resolution of Sentinel-2 satellite images to 2.5 meters using a set of neural network models. 

## **Installation**

Install the SEN2SRLite version using pip:

```bash
pip install mlstac sen2sr
```

For the full version, which use Mamba arquitecture, install as follows:

```bash
pip install mlstac sen2sr[full]
```


## Predict 10m and 20m bands


This example demonstrates the use of the `SEN2SRLite` model to enhance the spatial resolution of Sentinel-2 imagery. A 
Sentinel-2 L2A data cube is created over a specified region and time range using the cubo library, including both 10 m 
and 20 m bands. The pretrained model, downloaded via mlstac, takes a single normalized sample as input and predicts a 
HR output. The visualization compares the original RGB composite to the super-resolved result.


```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import cubo

import sen2sr
import mlstac

# Download the model
mlstac.download(
  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
  output_dir="model/SEN2SRLite",
)

# Load the model
model = mlstac.load("model/SEN2SRLite").compiled_model()
model = model.to(device)


# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=64,
    resolution=10
)


# Prepare the data to be used in the model, select just one sample 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Apply model
superX = model(X[None]).squeeze(0)

# Visualize the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(X[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[0].set_title("Original S2")
ax[1].imshow(superX[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()*4)
ax[1].set_title("Enhanced Resolution S2")
plt.show()
```

<p align="center">
  <img src="assets/images/first_plot.png" width="100%">
</p>


## **Predict only RGBNIR bands**

This example demonstrates the use of the SEN2SRLite_RGBN model variant to enhance the spatial resolution 
of only the 10 m Sentinel-2 bands: red (B04), green (B03), blue (B02), and near-infrared (B08). A Sentinel-2 L2A 
data cube is created using the cubo library for a specific location and date range. The input is normalized and 
passed to a pretrained non-reference model optimized for RGB+NIR inputs. 


```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import cubo

import sen2sr
import mlstac

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02", "B08"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=64,
    resolution=10
)


# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)


# Download the model
#mlstac.download(
#  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
#  output_dir="model/SEN2SRLite_RGBN",
#)

# Load the model
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model()
model = model.to(device)

# Apply model
superX = model(X[None]).squeeze(0)
```

## **Predict on large images**

This example demonstrates the use of `SEN2SRLite_RGBN` for super-resolving large Sentinel-2 RGB+NIR images by chunking the 
input into smaller overlapping tiles. Although the model is trained to operate on fixed-size 128×128 patches, the 
`sen2sr.predict_large` utility automatically segments larger inputs into these tiles, applies the model to 
each tile independently, and then reconstructs the full image. An overlap margin (e.g., 32 pixels) is 
introduced between tiles to minimize edge artifacts and ensure continuity across tile boundaries.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import cubo

import sen2sr
import mlstac

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02", "B08"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=1024,
    resolution=10
)


# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Load the model
#mlstac.download(
#  file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
#  output_dir="model/SEN2SRLite_RGBN",
#)
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model()

# Apply model
superX = sen2sr.predict_large(
    model=model,
    X=X, # The input tensor
    overlap=32, # The overlap between the patches
)
```


### Estimate the Local Attention Map of the model 📊

This example computes the Local Attention Map (LAM) to analyze the model's spatial sensitivity 
and robustness. The input image is scanned with a sliding window, and the model's attention is 
estimated across multiple upscaling factors. The resulting KDE map highlights regions where 
the model focuses more strongly, while the robustness vector quantifies the model's stability 
to spatial perturbations.


```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import cubo

import sen2sr
import mlstac

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02", "B08"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=128,
    resolution=10
)


# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)


kde_map, complexity_metric, robustness_metric, robustness_vector = sen2sr.lam(
    X=X.cpu(), # The input tensor
    model=model.srx4, # The SR model
    h=240, # The height of the window
    w=240, # The width of the window
    window=128, # The window size
    scales = ["2x", "3x", "4x", "5x", "6x"]
)

# Visualize the results
plt.imshow(kde_map)
plt.title("Kernel Density Estimation")
plt.show()

plt.plot(robustness_vector)
plt.title("Robustness Vector")
plt.show()
```

<p align="center">
  <img src="assets/images/kernel.png" width="50%">
</p>
<br>
<p align="center">
  <img src="assets/images/vector.png" width="70%">
</p>
