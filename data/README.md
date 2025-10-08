# Data Directory

This folder is intentionally left empty.  
No raw datasets are bundled due to license and size.  
Users must **preprocess** their own datasets into the format below.

---

## Required Layout

Each dataset must live in its own subfolder and include exactly two CSV files:

data/<dataset_name>/
├── bus.csv
└── branch.csv

Update `configs/default.yaml` accordingly:
```yaml
data:
  dataset: "<dataset_name>"
  data_dir: "data"
  node_file: "bus.csv"
  link_file: "branch.csv"
```

---

1) Bus file (bus.csv)

Required columns
- Node: bus identifier (string or integer).
- Recommended: unique per bus.
- If not unique: all rows having the same Node value will be merged into one bus.
- In that case, Facility_name is chosen arbitrarily from among the merged rows (no guarantee which one).
- Facility_name: facility name (string).
- Column name must be exactly Facility_name.
- Used by the searching (geocoding) step.


Example

Node,Facility_name
1,Substation_A
1,Sub_A_variant   <-- same Node -> will be merged; Facility_name picked arbitrarily
2,Substation_B
A,Plant_X


---

2) Branch file (branch.csv)

Required columns
- Node1: identifier of the first bus (must appear in bus.csv:Node)
- Node2: identifier of the second bus (must appear in bus.csv:Node)

Attributes
- Any additional edge attributes are allowed; column names are not restricted
(e.g., Length, Resistance, Reactance, Susceptance, Conductance, etc.).

Duplicate-edge merging
- If multiple rows represent the same unordered pair (e.g., 1,2 and 2,1), they are merged into one edge.
- Edge attributes are averaged across duplicates.

Example

Node1,Node2,Resistance,Reactance
1,2,0.01,0.05
2,1,0.03,0.07   <-- same pair as (1,2) -> will be merged by averaging attributes

Result of merging:

Node1,Node2,Resistance,Reactance
1,2,0.02,0.06

---

## Constraints & Notes
- Every branch.csv endpoint (Node1, Node2) must exist in bus.csv:Node.
- Facility_name in bus.csv is mandatory (used for geocoding); it is not used for node merging.
- When Node is non-unique, rows are merged by Node only; Facility_name is picked arbitrarily from those rows.
- Large/proprietary raw data should not be committed to the repo.

## Minimal Workflow
  1.	Convert your dataset into:

data/<dataset_name>/bus.csv
data/<dataset_name>/branch.csv


  2.	Set configs/default.yaml:

data:
  dataset: "<dataset_name>"
  data_dir: "data"
  node_file: "bus.csv"
  link_file: "branch.csv"

  3. The framework includes a simple geocoding function based on open libraries.  
   No external API key or Google Geocoding service is required.
	4.	Run:

python test.py --config configs/default.yaml


  5.	Outputs with reconstructed coordinates will be written to results/<experiment_name>/.