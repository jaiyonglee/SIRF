# sirf_code/search.py
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

import networkx as nx


def _normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        s = x.strip()
        try:
            f = float(s)
            return str(int(f)) if f.is_integer() else s
        except Exception:
            return s
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(x)) if x.is_integer() else str(x)
    return str(x).strip()


def _read_bus_branch(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    dataset = cfg["data"]["dataset"]
    data_dir = Path(cfg["data"]["data_dir"]).resolve()
    bus_file = cfg["data"].get("node_file", "bus.csv")
    branch_file = cfg["data"].get("link_file", "branch.csv")

    bus_path = data_dir / dataset / bus_file
    branch_path = data_dir / dataset / branch_file

    bus = pd.read_csv(bus_path)
    branch = pd.read_csv(branch_path)

    if "Facility_name" not in bus.columns:
        raise ValueError("bus.csv must contain a 'Facility_name' column.")
    if "Node" not in bus.columns:
        raise ValueError("bus.csv must contain a 'Node' column.")
    if "Node1" not in branch.columns or "Node2" not in branch.columns:
        raise ValueError("branch.csv must contain 'Node1' and 'Node2' columns.")

    bus["Node"] = bus["Node"].map(_normalize_id)
    branch["Node1"] = branch["Node1"].map(_normalize_id)
    branch["Node2"] = branch["Node2"].map(_normalize_id)

    return bus, branch, data_dir


def _geocode_bus(bus: pd.DataFrame, country: str) -> pd.DataFrame:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    from tqdm import tqdm

    suffix = f", {country}" if country else ""

    geolocator = Nominatim(user_agent="sirf-repro")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    unique_names = bus["Facility_name"].dropna().unique()
    total_unique = len(unique_names)
    total_rows   = len(bus)
    print(f"  Geocoding {total_unique} unique Facility_name(s) "
          f"(skipping {total_rows - total_unique} duplicate rows) ...")

    geo_records = []
    failed = 0

    with tqdm(
        unique_names,
        total=total_unique,
        desc="  Geocoding",
        unit="name",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  {percentage:3.0f}%  [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for name in pbar:
            query = str(name) + suffix
            loc = geocode(query)
            if loc is None:
                failed += 1
            geo_records.append({
                "Facility_name":   name,
                "Latitude_truth":  loc.latitude  if loc else None,
                "Longitude_truth": loc.longitude if loc else None,
            })
            pbar.set_postfix(resolved=total_unique - failed, not_found=failed)

    print(f"  Done — {total_unique - failed}/{total_unique} resolved, "
          f"{failed} not found.")

    geo_df = pd.DataFrame(geo_records)
    bus = bus.merge(geo_df, on="Facility_name", how="left")
    return bus


def _dedup_branch_average(branch: pd.DataFrame) -> pd.DataFrame:
    branch = branch[branch["Node1"] != branch["Node2"]].copy()

    numeric_cols = [
        c for c in branch.columns
        if c not in ["Node1", "Node2"] and pd.api.types.is_numeric_dtype(branch[c])
    ]

    duplicate_means = (
        branch.groupby(["Node1", "Node2"], as_index=True)[numeric_cols]
        .mean()
        .reset_index()
    )

    branch_nodup = branch.drop_duplicates(subset=["Node1", "Node2"]).copy()
    branch_merged = branch_nodup.merge(
        duplicate_means, on=["Node1", "Node2"], how="left", suffixes=("", "_mean")
    )

    for c in numeric_cols:
        if c + "_mean" in branch_merged.columns:
            branch_merged[c] = branch_merged[c + "_mean"]
            branch_merged.drop(columns=[c + "_mean"], inplace=True)

    return branch_merged.reset_index(drop=True)


def _filter_largest_components(branch: pd.DataFrame, bus: pd.DataFrame, component_n: int = 1):
    if branch.empty:
        return nx.Graph(), bus.iloc[0:0].copy(), branch.copy()

    G = nx.from_pandas_edgelist(branch, source="Node1", target="Node2")
    clusters = sorted(nx.connected_components(G), key=len, reverse=True)

    nodes_to_keep = set()
    for comp in clusters[: max(1, int(component_n))]:
        nodes_to_keep.update(comp)

    Gf = G.subgraph(nodes_to_keep).copy()
    bus_f = bus[bus["Node"].isin(Gf.nodes())].copy()
    branch_f = branch[
        branch["Node1"].isin(Gf.nodes()) & branch["Node2"].isin(Gf.nodes())
    ].copy()

    return Gf, bus_f.reset_index(drop=True), branch_f.reset_index(drop=True)


def run(cfg: Dict):
    bus, branch, data_dir = _read_bus_branch(cfg)
    branch = branch.dropna(subset=["Node1", "Node2"]).copy()

    country = cfg.get("searching", {}).get("country", "")
    bus = _geocode_bus(bus, country=country)

    branch = _dedup_branch_average(branch)

    component_n = cfg.get("searching", {}).get("component_n", 1)
    _, bus_f, branch_f = _filter_largest_components(branch, bus, component_n=component_n)

    dataset = cfg["data"]["dataset"]
    results_root = data_dir.parent / "results" / dataset / "Searched_data"
    results_root.mkdir(parents=True, exist_ok=True)

    bus_f = bus_f.drop(columns=["Address", "Location"], errors="ignore")
    bus_f.to_csv(results_root / "bus.csv", index=False)
    branch_f.to_csv(results_root / "branch.csv", index=False)

    return bus_f, branch_f