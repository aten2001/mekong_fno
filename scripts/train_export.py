# scripts/train_export.py
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")      # deterministic ops
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")    # adapt for older TensorFlow versions

import json
import sys
import time
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# --- make paths stable regardless of current working directory ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import TenYearUnifiedRunner, build_climatology_from_train_years

# set unified random seed
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def _resolve_under_repo(p: str) -> Path:
    """Resolve path as absolute; relative paths are interpreted under repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)

def parse_args():
    p = argparse.ArgumentParser("Train FNO and export four artifacts")

    p.add_argument(
        "--csv_dir",
        type=str,
        default=os.environ.get("CSV_DIR", str(REPO_ROOT / "data")),
        help="Raw CSV directory"
    )
    p.add_argument("--seq_length", type=int, default=120, help="Input window length L")
    p.add_argument("--pred_length", type=int, default=7, help="Prediction horizon H (should stay 7)")
    p.add_argument("--start_year", type=int, default=2015, help="Data start year (inclusive)")
    p.add_argument("--end_year",   type=int, default=2025, help="Data end year (inclusive)")
    p.add_argument("--epochs",     type=int, default=300, help="Max training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--tau_years",  type=float, default=12.0, help="Time-decay tau (years)")
    p.add_argument("--allow_missing_u", action="store_true", help="Allow missing discharge and fill u")

    p.add_argument(
        "--artifacts_dir",
        type=str,
        default=os.environ.get("ARTIFACTS_DIR", str(REPO_ROOT / "artifacts")),
        help="Artifacts output dir"
    )
    p.add_argument(
        "--weights_dir",
        type=str,
        default=os.environ.get("WEIGHTS_DIR", str(REPO_ROOT / "weights")),
        help="Weights output dir"
    )

    p.add_argument("--ckpt_name", type=str, default="stung_treng_fno.ckpt", help="Checkpoint prefix/name")
    p.add_argument("--k_phase",   type=int, default=10, help="Phase scan range K (±K)")
    return p.parse_args()

def main():
    args = parse_args()

    # 0) Resolve all user paths to stable absolute paths
    csv_dir = _resolve_under_repo(args.csv_dir)
    artifacts_dir = _resolve_under_repo(args.artifacts_dir)
    weights_dir = _resolve_under_repo(args.weights_dir)

    # 1) Validate output paths first
    for d in [artifacts_dir, weights_dir]:
        if d.exists() and not d.is_dir():
            print(f"Error: {d} exists but is not a directory!")
            print("Please remove or rename that file, then rerun.")
            sys.exit(1)

    # 2) Create directories
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=== Configuration ===")
    print(json.dumps({
        "csv_dir": str(csv_dir),
        "seq_length": args.seq_length,
        "pred_length": args.pred_length,
        "years": [args.start_year, args.end_year],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tau_years": args.tau_years,
        "allow_missing_u": args.allow_missing_u,
        "artifacts_dir": str(artifacts_dir),
        "weights_dir": str(weights_dir),
        "ckpt_name": args.ckpt_name,
        "k_phase": args.k_phase
    }, indent=2))

    # 3) Load data (remove leap days)
    runner = TenYearUnifiedRunner(str(csv_dir), seq_length=args.seq_length, pred_length=args.pred_length)
    data = runner.load_range_data(args.start_year, args.end_year, allow_missing_u=args.allow_missing_u)
    if len(data) == 0:
        raise RuntimeError("No usable days found. Check CSV_DIR and filenames.")

    # 4) Compute and inject the climatology (using training years 2015–2022)
    clim_vec = build_climatology_from_train_years(data, train_years=runner.train_years)
    runner.set_climatology(clim_vec)

    # 5) Build samples
    X, Y, tgt_season, tgt_dates = runner.prepare_sequences_no_season(data)
    print(f"Samples built: X={X.shape}, Y={Y.shape}")

    # 6) Train
    t0 = time.time()
    hist = runner.train_with_dual_window_val(
        X, Y, tgt_season, tgt_dates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_time_decay=True,
        tau_years=args.tau_years
    )
    print(f"Training finished in {(time.time() - t0)/60.0:.1f} min")

    # 7) evaluation: this prints the Val/Test window metrics plus the 2024 weighted metrics
    eval_res = runner.evaluate_all()

    # 8) Phase/amplitude diagnosis: compute once and reuse later when writing to disk
    print("\n=== Phase/Amplitude attribution: search k on 2023 and fix on 2024 as a transparent baseline ===")
    phase_report = runner.phase_vs_amplitude_report(K=args.k_phase)

    # 9) Export
    # 9.1 Export the weights
    ckpt_prefix = weights_dir / args.ckpt_name
    runner.model.save_weights(str(ckpt_prefix))
    print(f"Saved weights to prefix: {ckpt_prefix}*")

    # 9.2 Export the climatology baseline
    clim_path = artifacts_dir / "clim_vec.npy"
    np.save(str(clim_path), runner.clim)
    print(f"Saved climatology: {clim_path}")

    # 9.3 Export normalized statistics
    norm_path = artifacts_dir / "norm_stats.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(runner.norm_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved norm stats: {norm_path}")

    # 9.4 Export phase report
    phase_path = artifacts_dir / "phase_report.json"
    with open(phase_path, "w", encoding="utf-8") as f:
        json.dump(phase_report, f, ensure_ascii=False, indent=2)
    print(f"Saved phase report: {phase_path}")

    # 9.5 Export metrics (trimmed)
    def _trim_metrics(d):
        return {
            "h_rmse": float(d.get("h_rmse", float("nan"))),
            "h_mae": float(d.get("h_mae", float("nan"))),
            "n": int(d.get("n", 0)),
            "title": str(d.get("title", "")),
        }

    eval_res_small = {k: _trim_metrics(v) for k, v in eval_res.items()}
    metrics_path = artifacts_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_res_small, f, ensure_ascii=False, indent=2)
    print(f"Saved eval metrics: {metrics_path}")

    # 9.6 Export training curves
    hist_path = artifacts_dir / "train_history.json"
    try:
        h = hist.history if hasattr(hist, "history") else {}
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, indent=2)
        print(f"Saved train history: {hist_path}")
    except Exception as e:
        print("Warn: failed to save train history:", e)

    print("\nExported artifacts:")
    print(" - Weights:", str(ckpt_prefix), "[.index, .data-00000-of-00001]")
    print(" - Artifacts:", str(clim_path), str(norm_path), str(phase_path), str(metrics_path), str(hist_path))

if __name__ == "__main__":
    main()
