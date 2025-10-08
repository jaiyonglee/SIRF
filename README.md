# Spatial Information Reconstruction Framework (SIRF)

This repository provides the official implementation for reproducing the results of:

**Spatial information reconstruction framework for power grid datasets without geographic coordinates**  
Jaiyong Lee[1], Daekyung Lee[1,2], Heetae Kim[1,†]  
1 Department of Energy Technology, Korea Institute of Energy Technology (KENTECH)  
2 Supply Chain Intelligence Institute Austria, Vienna, Austria  

† hkim@kentech.ac.kr  

Publish*ing* in *PLoS ONE*, 2025  

---

## Overview
This repository contains the source code, configuration files, and demo dataset to reproduce the experiments presented in the paper.  
The framework (SIRF) reconstructs missing geographic coordinates of power grid datasets that contain network topology and electrical properties but lack spatial information.  

The implementation consists of two main procedures:
1. **Searching procedure** – Extract facility names and retrieve geographic coordinates from open data sources (implemented here with the `geopy` library and OpenStreetMap Nominatim).
2. **Inference procedure** – Apply spatial inference through coordinate initialization, regression-based distance estimation, gradient-based optimization, anchor node reclassification, and dangling node repositioning.  
   *(This implementation runs a single-pass, non-ensemble pipeline; no coordinate averaging step is included.)*

---

## Requirements
- Python >= 3.9  
- Conda environment recommended

Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate sirf-env
```


---

## Dataset

Demo dataset
- A small demo dataset is provided under data/demo/ (Seoul, Suwon, Gwangju, Wonju, etc.).
This dataset allows users to test the framework out-of-the-box.

Research datasets (Not included here)
- French Grid (RTE, 2019) – Static grid model
- Danish Grid (Energinet, 2020) – Balanced load flow case

Both real-world datasets are publicly available but originally lack geographic coordinates.
Instructions for preprocessing datasets are provided in data/README.md.

---

## Usage

Run the reconstruction framework with a configuration file:

python sirf.py --config configs/default.yaml

Example (ablation study):

python sirf.py --config configs/ablation1.yaml

Results will be stored under results/<dataset>/.
- If all bus nodes are successfully geocoded during the searching procedure, the framework saves outputs immediately and skips inference.
- If some nodes remain without coordinates, the inference procedure runs automatically.

---

## Results

Example outputs include:
- results/demo/bus.csv / results/demo/branch.csv (final outputs)
- Intermediate files under results/<dataset>/searched/, .../Inferred_data/, .../Aligned_data/

File format:
- bus.csv: Node identifiers with reconstructed geographic coordinates (latitude, longitude).
- branch.csv: Transmission line connectivity and associated attributes.

---

## Demo Visualization
```
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data
bus = pd.read_csv("results/demo/bus.csv")
branch = pd.read_csv("results/demo/branch.csv")

# Normalize column names
bus = bus.rename(columns={
    "Latitude_truth": "Latitude",
    "Longitude_truth": "Longitude"
})

# Create figure with geographic projection
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add base map (land, borders, coastlines, lakes, rivers)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.3)
ax.add_feature(cfeature.RIVERS, alpha=0.3)

# Plot nodes
ax.scatter(bus['Longitude'], bus['Latitude'],
           s=30, c='blue', label='Nodes', transform=ccrs.PlateCarree())

# Plot branches (connections)
for _, row in branch.iterrows():
    n1 = bus[bus['Node'] == row['Node1']].iloc[0]
    n2 = bus[bus['Node'] == row['Node2']].iloc[0]
    ax.plot([n1['Longitude'], n2['Longitude']],
            [n1['Latitude'], n2['Latitude']],
            'k-', alpha=0.5, transform=ccrs.PlateCarree())

# Title and legend
ax.set_title("Reconstructed Grid (Demo)", fontsize=12)
ax.legend(loc='lower left')

# Automatically adjust view to data region
lon_min, lon_max = bus['Longitude'].min(), bus['Longitude'].max()
lat_min, lat_max = bus['Latitude'].min(), bus['Latitude'].max()
ax.set_extent([lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1],
              crs=ccrs.PlateCarree())

plt.tight_layout()
plt.show()
```

---

## Citation

If you use this repository, please cite our paper:

@article{lee2025sirf,
  title   = {Spatial information reconstruction framework for power grid datasets without geographic coordinates},
  author  = {Jaiyong Lee, Daekyung Lee, and Heetae Kim},
  journal = {PLoS ONE},
  year    = {2025}
}


---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
