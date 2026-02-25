# Spatial Information Reconstruction Framework (SIRF)

This repository provides the official implementation for reproducing the results of:

**Spatial information reconstruction framework for power grid datasets without geographic coordinates**  
Jaiyong Lee[1], Daekyung Lee[1,2], Heetae Kim[1,†]  
1 Department of Energy Technology, Korea Institute of Energy Technology (KENTECH)  
2 Supply Chain Intelligence Institute Austria, Vienna, Austria  

† hkim@kentech.ac.kr  

---

## Overview
This repository contains the source code, configuration files, and demo dataset to reproduce the experiments presented in the paper.  
The framework (SIRF) reconstructs missing geographic coordinates of power grid datasets that contain network topology, facility names, and line parameters, but lack coordinate information.  

The implementation consists of two main procedures:
1. **Searching procedure** – Extract facility names and retrieve geographic coordinates from open data sources (implemented here with the `geopy` library and OpenStreetMap `Nominatim`). Note that our results were obtained using the `Google Maps Geocoding API`, which may yield higher accuracy than the default Nominatim implementation.
2. **Inference and Refining procedure** – Apply spatial inference through coordinate initialization, regression-based distance estimation, gradient-based optimization, anchor node reclassification, and dangling node repositioning.  
   *(This implementation runs a single-pass, non-ensemble pipeline; no coordinate averaging step is included.)*

---

## Requirements
- Python >= 3.11
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

Example (dataset_ex):

python sirf.py --config configs/dataset_ex.yaml

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

## Citation

If you use this repository, please cite our paper:

@article{sirf,
  title   = {Spatial information reconstruction framework for power grid datasets without geographic coordinates},
  author  = {Jaiyong Lee, Daekyung Lee, and Heetae Kim},
  journal = {},
  year    = {2026}
}


---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
