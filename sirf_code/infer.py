# sirf_code/infer.py
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# ==== Library imports (moved) ====
from sirf_code.library.Data_Scaling import control_retention_rate, copy_position_to_link_data
from sirf_code.library.Linear_Regression import regress_link_data
from sirf_code.library.Initialize import initialize_position
from sirf_code.library.Inference import infer_position, filter_data
from sirf_code.library.Align_Node import align_dangling_nodes
# =================================


# ---------- small IO helpers ----------
def _res_root(cfg: Dict[str, Any]) -> Path:
    """results/<dataset_name> root based on cfg.data.data_dir"""
    data_dir = Path(cfg["data"]["data_dir"]).resolve()
    dataset = cfg["data"]["dataset"]
    return data_dir.parent / "results" / dataset

def _ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _save_csv(df: pd.DataFrame, res_root: Path, rel: str) -> Path:
    p = _ensure_parent(res_root / rel)
    df.to_csv(p, index=False)
    return p

def _save_array(arr: np.ndarray, res_root: Path, rel: str) -> Path:
    p = _ensure_parent(res_root / rel)
    arr.tofile(p, sep=",")
    return p
# --------------------------------------


def _get_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read params from YAML (configs). Provide safe defaults."""
    inf = cfg.get("inference", {})
    return {
        "retention_rate": float(inf.get("retention_rate", 1.0)),
        "iter_num": int(inf.get("iter_num", 0)),
        "regression_poly_degree": int(inf.get("regression_poly_degree", 2)),
        "iter_n": int(inf.get("iter_n", 500)),
        "filter_n": int(inf.get("filter_n", 0)),
        "num_seed": int(inf.get("num_seed", 42)),
        "include_bias": bool(inf.get("include_bias", True)),
    }


def run(bus: pd.DataFrame, branch: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standalone, config-driven inference pipeline.
    - Read parameters from YAML (`cfg['inference']`).
    - Use in-memory `bus`/`branch` given by search.run(cfg).
    - Save all artifacts under: results/<dataset_name>/**.
    - Return final aligned bus/branch.
    """
    logging.info("[infer] starting pipeline (config-driven, non-ensemble).")
    P = _get_params(cfg)
    res = _res_root(cfg)

    # Common suffix like original code: e.g., 80p_0001
    suf = f"{round(P['retention_rate']*100)}p_{str(P['iter_num']).zfill(4)}"

    # ======================
    # 1) Data Scaling
    # ======================
    # Note: original code tried to read pre-filtered CSVs; here we use in-memory inputs.
    df_bus_scaled, df_bus_nan = control_retention_rate(
        bus, retention_rate=P["retention_rate"], num_seed=P["num_seed"]
    )
    df_branch_detailed = copy_position_to_link_data(df_bus_scaled, branch)

    _save_csv(df_bus_scaled, res, f"Preprocessed_data/Scaled_position/Bus_scaled_{suf}.csv")
    _save_csv(df_bus_nan,    res, f"Preprocessed_data/Nan_position/Bus_nan_{suf}.csv")
    _save_csv(df_branch_detailed, res, f"Preprocessed_data/Detailed_branch/Branch_detailed_{suf}.csv")

    # Free early
    del df_bus_scaled, df_bus_nan, df_branch_detailed

    # ======================
    # 2) Linear Regression
    # ======================
    df_bus_scaled = pd.read_csv(res / f"Preprocessed_data/Scaled_position/Bus_scaled_{suf}.csv")
    df_bus_nan    = pd.read_csv(res / f"Preprocessed_data/Nan_position/Bus_nan_{suf}.csv")
    df_branch_detailed = pd.read_csv(res / f"Preprocessed_data/Detailed_branch/Branch_detailed_{suf}.csv")

    df_branch_regressed, df_coef = regress_link_data(
        df_branch_detailed,
        regression_poly_degree=P["regression_poly_degree"],
        include_bias=P["include_bias"],
    )

    _save_csv(df_branch_regressed, res, f"Regressed_data/Regressed_branch/Branch_regressed_{suf}.csv")
    _save_csv(df_coef,             res, f"Regressed_data/Model/Model_regressed_{suf}.csv")

    # Free early
    del df_branch_detailed, df_branch_regressed, df_coef

    # ======================
    # 3) Initialization
    # ======================
    df_bus_scaled = pd.read_csv(res / f"Preprocessed_data/Scaled_position/Bus_scaled_{suf}.csv")
    df_branch_regressed = pd.read_csv(res / f"Regressed_data/Regressed_branch/Branch_regressed_{suf}.csv")

    df_bus_initialized, df_branch_initialized = initialize_position(df_bus_scaled, df_branch_regressed)

    _save_csv(df_bus_initialized,   res, f"Initialized_data/Bus_initialized/Bus_initialized_{suf}.csv")
    _save_csv(df_branch_initialized, res, f"Initialized_data/Branch_initialized/Branch_initialized_{suf}.csv")

    del df_bus_scaled, df_branch_regressed, df_bus_initialized, df_branch_initialized

    # ======================
    # 4) Position Inference (+ optional iterative filtering)
    # ======================
    df_bus_initialized   = pd.read_csv(res / f"Initialized_data/Bus_initialized/Bus_initialized_{suf}.csv")
    df_branch_initialized = pd.read_csv(res / f"Initialized_data/Branch_initialized/Branch_initialized_{suf}.csv")
    df_bus_nan           = pd.read_csv(res / f"Preprocessed_data/Nan_position/Bus_nan_{suf}.csv")

    df_bus_inferred, df_branch_inferred, outputs = infer_position(
        df_bus_initialized,
        df_branch_initialized,
        df_bus_nan,
        outputs=[],
        iter_n=P["iter_n"],
    )

    # Iterative filtering (optional)
    loss_per_node_changes: List[float] = []
    node_to_add_list: List[Any] = []

    if P["filter_n"] > 0:
        # reload original branch if needed by filter_data (keep semantics similar to legacy)
        # Here we use the *input* branch given to run(), not from disk.
        for i in range(P["filter_n"]):
            logging.info(f"[infer] filtering+reinfer iteration {i+1}/{P['filter_n']}")
            df_bus_nan, node_to_add = filter_data(df_bus_inferred, df_branch_inferred, df_bus_nan, outputs)

            # re-regress with updated inferred branch
            df_branch_regressed, df_coef = regress_link_data(
                df_branch_inferred,
                regression_poly_degree=P["regression_poly_degree"],
                include_bias=P["include_bias"],
            )
            # replace predicted distance
            df_branch_inferred["Predicted_distance"] = df_branch_regressed["Predicted_distance"]

            # re-infer
            df_bus_inferred, df_branch_inferred, outputs = infer_position(
                df_bus_inferred,
                df_branch_inferred,
                df_bus_nan,
                outputs=outputs,
                iter_n=P["iter_n"],
            )
            loss_per_node_changes.append(outputs["loss"][-1])
            node_to_add_list.append(node_to_add)

        # Save (re)regressed artifacts too, mirroring legacy layout
        _save_csv(df_branch_regressed, res, f"Regressed_data/Regressed_branch/Branch_regressed_{suf}.csv")
        _save_csv(df_coef,             res, f"Regressed_data/Model/Model_regressed_{suf}.csv")

    # Save inferred artifacts
    _save_csv(df_bus_inferred,    res, f"Inferred_data/Bus_inferred/Bus_inferred_{suf}.csv")
    _save_csv(df_branch_inferred, res, f"Inferred_data/Branch_inferred/Branch_inferred_{suf}.csv")

    if loss_per_node_changes:
        _save_array(np.array(loss_per_node_changes, dtype=float), res, f"Inferred_data/Inference_loss/Inference_loss_{suf}.csv")
    if "distance_rate" in outputs:
        _save_array(np.array(outputs["distance_rate"]), res, f"Inferred_data/Distance_rate/Distance_rate_{suf}.csv")

    _save_csv(df_bus_nan, res, f"Inferred_data/Nan_position/Bus_nan_{suf}.csv")
    if node_to_add_list:
        _save_array(np.array(node_to_add_list, dtype=object), res, f"Inferred_data/Filtered_nodes/Filtered_nodes_{suf}.csv")

    del df_bus_initialized, df_branch_initialized
    del df_bus_inferred, df_branch_inferred, df_bus_nan, outputs

    # ======================
    # 5) Align Dangling Nodes
    # ======================
    df_bus_inferred  = pd.read_csv(res / f"Inferred_data/Bus_inferred/Bus_inferred_{suf}.csv")
    df_branch_inferred = pd.read_csv(res / f"Inferred_data/Branch_inferred/Branch_inferred_{suf}.csv")
    df_bus_nan       = pd.read_csv(res / f"Inferred_data/Nan_position/Bus_nan_{suf}.csv")

    df_bus_aligned, df_branch_aligned = align_dangling_nodes(
        df_bus_inferred, df_branch_inferred, df_bus_nan, n_squared=1
    )

    _save_csv(df_bus_aligned,   res, f"Aligned_data/Bus_aligned/Bus_aligned_{suf}.csv")
    _save_csv(df_branch_aligned, res, f"Aligned_data/Branch_aligned/Branch_aligned_{suf}.csv")

    logging.info("[infer] Position inference completed.")
    return df_bus_aligned, df_branch_aligned