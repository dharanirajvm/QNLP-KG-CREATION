import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/evaluation/timing metrics from a run directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            r"c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\inference_snapshots\quantum_fb15k237_20260308_174529_updated_20260310_193344"
        ),
        help="Run directory containing metrics_history.jsonl. Replace with your final run folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\outputs\metrics_viz"
        ),
        help="Directory to save charts and summaries.",
    )
    parser.add_argument("--smooth", type=int, default=3, help="Rolling window for smoothed curves.")
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSON rows found in: {path}")
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "epoch_seconds" in df.columns:
        df["cum_hours"] = df["epoch_seconds"].fillna(0).cumsum() / 3600.0
    else:
        df["cum_hours"] = 0.0
    if "train_samples" in df.columns and "epoch_seconds" in df.columns:
        denom = df["epoch_seconds"].replace(0, pd.NA)
        df["samples_per_sec"] = df["train_samples"] / denom
    return df


def maybe_smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_overview(df: pd.DataFrame, out_path: Path, smooth: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="train_loss")
    ax1.plot(df["epoch"], maybe_smooth(df["train_loss"], smooth), linestyle="--", linewidth=2, label=f"loss_ma{smooth}")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(df["epoch"], df["train_pair_acc"], marker="o", color="tab:green", linewidth=2, label="train_pair_acc")
    ax2.plot(
        df["epoch"],
        maybe_smooth(df["train_pair_acc"], smooth),
        linestyle="--",
        color="tab:olive",
        linewidth=2,
        label=f"pair_acc_ma{smooth}",
    )
    ax2.set_title("Training Pair Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pair Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    if "val_mrr" in df.columns:
        eval_df = df[df["val_mrr"].notna()]
        ax3.plot(eval_df["epoch"], eval_df["val_mrr"], marker="o", linewidth=2, label="val_mrr")
        if "val_hits@10" in eval_df.columns:
            ax3.plot(eval_df["epoch"], eval_df["val_hits@10"], marker="o", linewidth=2, label="val_hits@10")
        if "val_hits@3" in eval_df.columns:
            ax3.plot(eval_df["epoch"], eval_df["val_hits@3"], marker="o", linewidth=2, label="val_hits@3")
    ax3.set_title("Validation Ranking Metrics")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.grid(alpha=0.3)
    ax3.legend()

    if "epoch_seconds" in df.columns:
        ax4.bar(df["epoch"], df["epoch_seconds"] / 60.0, alpha=0.6, label="epoch_minutes")
    ax4_t = ax4.twinx()
    ax4_t.plot(df["epoch"], df["cum_hours"], color="tab:red", marker="o", linewidth=2, label="cumulative_hours")
    ax4.set_title("Training Time")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Minutes / epoch")
    ax4_t.set_ylabel("Cumulative hours")
    ax4.grid(alpha=0.3)
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4_t.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_eval_details(df: pd.DataFrame, out_path: Path) -> None:
    eval_df = df[df.get("val_mrr", pd.Series(dtype=float)).notna()].copy()
    if eval_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(eval_df["epoch"], eval_df["val_mr"], marker="o", linewidth=2, label="val_mr", color="tab:orange")
    axes[0].set_title("Validation MR (lower is better)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MR")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(eval_df["epoch"], eval_df["val_hits@1"], marker="o", linewidth=2, label="hits@1")
    axes[1].plot(eval_df["epoch"], eval_df["val_hits@3"], marker="o", linewidth=2, label="hits@3")
    axes[1].plot(eval_df["epoch"], eval_df["val_hits@10"], marker="o", linewidth=2, label="hits@10")
    axes[1].set_title("Validation Hits@K")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Hits")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_dir: Path) -> dict:
    summary = {
        "epochs_recorded": int(df["epoch"].max()),
        "total_training_hours": float(df["cum_hours"].iloc[-1]),
        "best_train_loss": float(df["train_loss"].min()),
        "best_train_loss_epoch": int(df.loc[df["train_loss"].idxmin(), "epoch"]),
        "best_train_pair_acc": float(df["train_pair_acc"].max()),
        "best_train_pair_acc_epoch": int(df.loc[df["train_pair_acc"].idxmax(), "epoch"]),
    }

    eval_df = df[df.get("val_mrr", pd.Series(dtype=float)).notna()].copy()
    if not eval_df.empty:
        idx_mrr = eval_df["val_mrr"].idxmax()
        idx_h3 = eval_df["val_hits@3"].idxmax()
        idx_h10 = eval_df["val_hits@10"].idxmax()
        idx_mr = eval_df["val_mr"].idxmin()
        summary.update(
            {
                "best_val_mrr": float(eval_df.loc[idx_mrr, "val_mrr"]),
                "best_val_mrr_epoch": int(eval_df.loc[idx_mrr, "epoch"]),
                "best_val_hits@3": float(eval_df.loc[idx_h3, "val_hits@3"]),
                "best_val_hits@3_epoch": int(eval_df.loc[idx_h3, "epoch"]),
                "best_val_hits@10": float(eval_df.loc[idx_h10, "val_hits@10"]),
                "best_val_hits@10_epoch": int(eval_df.loc[idx_h10, "epoch"]),
                "best_val_mr": float(eval_df.loc[idx_mr, "val_mr"]),
                "best_val_mr_epoch": int(eval_df.loc[idx_mr, "epoch"]),
            }
        )

    summary_path = out_dir / "metrics_summary_viz.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "metrics_history_table.csv", index=False, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics_history.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_history.jsonl not found in run directory: {run_dir}")

    df = load_metrics(metrics_path)

    plot_overview(df, out_dir / "training_metrics_overview.png", args.smooth)
    plot_eval_details(df, out_dir / "eval_metrics_detail.png")
    summary = write_summary(df, out_dir)

    print("Saved:")
    print("-", out_dir / "training_metrics_overview.png")
    print("-", out_dir / "eval_metrics_detail.png")
    print("-", out_dir / "metrics_history_table.csv")
    print("-", out_dir / "metrics_summary_viz.json")
    print("\nBest checkpoints:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
