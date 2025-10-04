# scripts/train_export.py
import os
import json
import time
import argparse
import numpy as np

from src.runner import TenYearUnifiedRunner, build_climatology_from_train_years

def parse_args():
    p = argparse.ArgumentParser("Train FNO and export four artifacts")
    p.add_argument("--csv_dir", type=str, default=os.environ.get("CSV_DIR", "."), help="Raw CSV directory")
    p.add_argument("--seq_length", type=int, default=120, help="Input window length L")
    p.add_argument("--pred_length", type=int, default=7, help="Prediction horizon H (should stay 7)")
    p.add_argument("--start_year", type=int, default=2015, help="Data start year (inclusive)")
    p.add_argument("--end_year",   type=int, default=2025, help="Data end year (inclusive)")
    p.add_argument("--epochs",     type=int, default=300, help="Max training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--tau_years",  type=float, default=12.0, help="Time-decay tau (years)")
    p.add_argument("--allow_missing_u", action="store_true", help="Allow missing discharge and fill u")
    p.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts output dir")
    p.add_argument("--weights_dir",   type=str, default="weights", help="Weights output dir")
    p.add_argument("--ckpt_name",     type=str, default="stung_treng_fno.ckpt", help="Checkpoint prefix/name")
    p.add_argument("--k_phase",       type=int, default=10, help="Phase scan range K (±K)")
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(args.weights_dir,   exist_ok=True)

    print("=== Configuration ===")
    print(json.dumps({
        "csv_dir": args.csv_dir,
        "seq_length": args.seq_length,
        "pred_length": args.pred_length,
        "years": [args.start_year, args.end_year],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tau_years": args.tau_years,
        "allow_missing_u": args.allow_missing_u,
        "artifacts_dir": args.artifacts_dir,
        "weights_dir": args.weights_dir,
        "ckpt_name": args.ckpt_name,
        "k_phase": args.k_phase
    }, indent=2))

    # 1) 加载原始数据（移除闰日）——允许缺失流量可用 --allow_missing_u
    runner = TenYearUnifiedRunner(args.csv_dir, seq_length=args.seq_length, pred_length=args.pred_length)
    data = runner.load_range_data(args.start_year, args.end_year, allow_missing_u=args.allow_missing_u)
    if len(data) == 0:
        raise RuntimeError("No usable days found. Check CSV_DIR and filenames.")

    # 2) 用训练年(2015–2022)计算逐日 climatology（doy: 1..365；index 0 为空位）
    clim_vec = build_climatology_from_train_years(data, train_years=runner.train_years)
    runner.set_climatology(clim_vec)

    # 3) 组装样本（特征6通道，目标为绝对 h 序列；训练时转为 anomaly）
    X, Y, tgt_season, tgt_dates = runner.prepare_sequences_no_season(data)
    print(f"Samples built: X={X.shape}, Y={Y.shape}")

    # 4) 训练（验证集=2023 干/湿两窗；测试集=2024 干/湿两窗）
    t0 = time.time()
    hist = runner.train_with_dual_window_val(
        X, Y, tgt_season, tgt_dates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_time_decay=True,
        tau_years=args.tau_years
    )
    t1 = time.time()
    print(f"Training finished in {(t1 - t0)/60.0:.1f} min")

    # 5) 评估（打印 RMSE/MAE）并导出“四件套”
    # 5.1 权重（注意：TF checkpoint 会生成 .index 和 .data-00000-of-00001 两个文件）
    ckpt_prefix = os.path.join(args.weights_dir, args.ckpt_name)
    runner.model.save_weights(ckpt_prefix)
    print(f"Saved weights to prefix: {ckpt_prefix}*")

    # 5.2 气候基线
    clim_path = os.path.join(args.artifacts_dir, "clim_vec.npy")
    np.save(clim_path, runner.clim)
    print(f"Saved climatology: {clim_path}")

    # 5.3 归一化统计
    norm_path = os.path.join(args.artifacts_dir, "norm_stats.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(runner.norm_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved norm stats: {norm_path}")

    # 5.4 相位报告（2023 扫 k* → 应用于 2024）
    phase_report = runner.phase_vs_amplitude_report(K=args.k_phase)
    phase_path = os.path.join(args.artifacts_dir, "phase_report.json")
    with open(phase_path, "w", encoding="utf-8") as f:
        json.dump(phase_report, f, ensure_ascii=False, indent=2)
    print(f"Saved phase report: {phase_path}")

    # 额外：保存训练曲线（loss）
    hist_path = os.path.join(args.artifacts_dir, "train_history.json")
    try:
        # Keras History -> dict(list)
        h = hist.history if hasattr(hist, "history") else {}
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, indent=2)
        print(f"Saved train history: {hist_path}")
    except Exception as e:
        print("Warn: failed to save train history:", e)

    print("\nExported artifacts:")
    print(" - Weights:", ckpt_prefix, "[.index, .data-00000-of-00001]")
    print(" - Artifacts:", clim_path, norm_path, phase_path, hist_path)

if __name__ == "__main__":
    main()
