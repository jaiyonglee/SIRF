# sirf_code/search.py
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

import networkx as nx


def _normalize_id(x) -> str:
    """
    Normalize node identifiers so that 1, '1', 1.0 -> '1'.
    """
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

    # Normalize IDs to strings like '1'
    bus["Node"] = bus["Node"].map(_normalize_id)
    branch["Node1"] = branch["Node1"].map(_normalize_id)
    branch["Node2"] = branch["Node2"].map(_normalize_id)

    return bus, branch, data_dir


def _geocode_bus(bus: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Simple Nominatim geocoding (no API key). Adds Latitude_truth/Longitude_truth.
    """
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    suffix = f", {country}" if country else ""
    bus["Address"] = bus["Facility_name"].astype(str) + suffix

    geolocator = Nominatim(user_agent="sirf-repro")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    bus["Location"] = bus["Address"].apply(lambda q: geocode(q))
    bus["Latitude_truth"] = bus["Location"].apply(lambda loc: loc.latitude if loc else None)
    bus["Longitude_truth"] = bus["Location"].apply(lambda loc: loc.longitude if loc else None)
    return bus


def _dedup_branch_average(branch: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop self-loops
    - Average duplicate edges (Node1,Node2). Treat as directed pairs here since
      상위 필터에서 무방향 묶음을 원하면 사전에 정렬해서 키를 만들면 됨.
      요청 주신 코드 흐름을 그대로 반영: groupby mean → 병합.
    """
    # Remove self-loops
    branch = branch[branch["Node1"] != branch["Node2"]].copy()

    # Numeric columns to mean, non-numeric keep first
    numeric_cols = [c for c in branch.columns if c not in ["Node1", "Node2"] and pd.api.types.is_numeric_dtype(branch[c])]
    # mean for duplicates
    duplicate_means = branch.groupby(["Node1", "Node2"], as_index=True)[numeric_cols].mean().reset_index()

    # Drop duplicate rows keeping first, then overwrite numeric cols with means
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
    """
    Keep nodes/edges belonging to the top-N largest connected components.
    Uses an undirected view of the graph.
    """
    if branch.empty:
        return nx.Graph(), bus.iloc[0:0].copy(), branch.copy()

    # Undirected graph from edges
    G = nx.from_pandas_edgelist(branch, source="Node1", target="Node2")

    # Sort components by size
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
    """
    Pipeline (simple, as requested):
      1) Read bus/branch (respect data_dir which may be '../data')
      2) Geocode bus using Facility_name + ', <country>'
      3) Branch: drop NA Node1/Node2 → remove self-loops → average duplicates
      4) Keep top-N largest components (component_n from cfg.searching.component_n, default 1)
      5) Save both to <data_dir>/../results/<dataset>/searched/{bus.csv, branch.csv}
      6) Return filtered bus (with Latitude_truth/Longitude_truth) and filtered branch
    """
    # 1) read
    bus, branch, data_dir = _read_bus_branch(cfg)

    # drop NaN Node1/Node2 rows early
    branch = branch.dropna(subset=["Node1", "Node2"]).copy()

    # 2) geocode
    country = cfg.get("searching", {}).get("country", "")
    bus = _geocode_bus(bus, country=country)

    # 3) branch cleanup
    branch = _dedup_branch_average(branch)

    # 4) top-N components
    component_n = cfg.get("searching", {}).get("component_n", 1)
    _, bus_f, branch_f = _filter_largest_components(branch, bus, component_n=component_n)

    # 5) save (results alongside data_dir)
    dataset = cfg["data"]["dataset"]
    results_root = data_dir.parent / "results" / dataset / "Searched_data"
    results_root.mkdir(parents=True, exist_ok=True)

    # bus: only Node + truth coords (요청 그대로)
    bus_out = results_root / "bus.csv"
    bus_f = bus_f.drop(columns=["Address", "Location"], errors="ignore")
    bus_f.to_csv(bus_out, index=False)

    # branch: 정제/필터링 결과 전체 저장
    branch_out = results_root / "branch.csv"
    branch_f.to_csv(branch_out, index=False)

    return bus_f, branch_f