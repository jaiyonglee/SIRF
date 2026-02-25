# sirf.py
import argparse
import logging
from pathlib import Path
import yaml
import sys
import pandas as pd

# local modules
from sirf_code import search, infer


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_outdir(cfg: dict) -> Path:
    data_dir = Path(cfg["data"]["data_dir"]).resolve()
    dataset = cfg["data"]["dataset"]
    out = data_dir.parent / "results" / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(cfg: dict):
    """Reusable entry point (importable)."""
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    out_dir = ensure_outdir(cfg)
    dataset = cfg["data"]["dataset"]

    # 1) Searching / preprocessing
    if cfg.get("searching", {}).get("enable", True):
        bus_df, branch_df = search.run(cfg)
    else:
        logging.info("[sirf] Skipping searching (searching.enable = false)")

        data_dir = Path(cfg["data"]["data_dir"]).resolve()
        dataset = cfg["data"]["dataset"]
        bus_file = cfg["data"].get("node_file", "bus.csv")
        branch_file = cfg["data"].get("link_file", "branch.csv")

        results_root = data_dir.parent / "results" / dataset / "Searched_data"
        results_root.mkdir(parents=True, exist_ok=True)

        bus_path = results_root / "bus.csv"
        branch_path = results_root / "branch.csv"

        # 1) 이미 있으면 읽기
        if bus_path.exists() and branch_path.exists():
            bus_df = pd.read_csv(bus_path)
            branch_df = pd.read_csv(branch_path)

        # 2) 없으면 "원본 입력"을 읽어서 동일 이름/경로로 저장 (Facility_name 없어도 OK)
        else:
            logging.info("[sirf] Searched_data files not found. Creating them by copying raw inputs.")
            raw_bus_path = data_dir / dataset / bus_file
            raw_branch_path = data_dir / dataset / branch_file

            bus_df = pd.read_csv(raw_bus_path)
            branch_df = pd.read_csv(raw_branch_path)

            bus_df.to_csv(bus_path, index=False)
            branch_df.to_csv(branch_path, index=False)

    # === 결측치 체크 ===
    if bus_df[["Latitude_truth", "Longitude_truth"]].notna().all().all():
        bus_out, branch_out = bus_df, branch_df
        bus_out.to_csv(out_dir / "bus.csv", index=False)
        branch_out.to_csv(out_dir / "branch.csv", index=False)
        with open(out_dir / "logs.txt", "a") as f:
            f.write("Completed experiment (no inference needed: no missing values)\n")
        logging.info("No missing coordinates. Inference skipped.")
        return bus_out, branch_out

    # 2) Inference
    if cfg.get("inference", {}).get("enable", True):
        bus_out, branch_out = infer.run(bus_df, branch_df, cfg)
    else:
        logging.info("[sirf] Skipping inference (inference.enable = false)")
        bus_out, branch_out = bus_df, branch_df

    # 3) Minimal log
    with open(out_dir / "logs.txt", "a") as f:
        f.write(f"Completed experiment={dataset}\n")

    logging.info(f"Saved: {out_dir / 'bus.csv'}")
    logging.info(f"Saved: {out_dir / 'branch.csv'}")
    logging.info(f"Logs:  {out_dir / 'logs.txt'}")

    return bus_out, branch_out


def main():
    parser = argparse.ArgumentParser(description="SIRF Orchestrator")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",   # 기본값 설정
        help="Path to YAML config (default: configs/default.yaml)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()