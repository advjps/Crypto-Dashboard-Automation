# charts_analytics.py
import os
from datetime import datetime
import pytz
import pandas as pd
import matplotlib.pyplot as plt

ANALYTICS_DIR = "analytics"
ALL_SIGNALS_CSV = os.path.join(ANALYTICS_DIR, "all_signals.csv")

BUCKET_ORDER = ["Strong Buy", "Buy", "Strong Sell", "Sell"]

def ist_stamp():
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def winrate(success, fail):
    denom = (success or 0) + (fail or 0)
    return (success / denom * 100.0) if denom > 0 else 0.0

def load_all():
    if not os.path.exists(ALL_SIGNALS_CSV):
        raise FileNotFoundError(f"{ALL_SIGNALS_CSV} not found. Run merge_analytics.py first.")
    df = pd.read_csv(ALL_SIGNALS_CSV)
    # Normalize types
    for col in ["Confidence","Num_Conf","ADX15m","MACD_Hist","PercentB","RegimeScore",
                "DeservedStrongBuy","DeservedStrongSell"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def plot_winrate_by_bucket(df):
    stamp = ist_stamp()
    buckets = []
    wrs = []
    counts = []
    for b in BUCKET_ORDER:
        sub = df[df["Signal"] == b]
        succ = (sub["Outcome"] == "Success").sum()
        fail = (sub["Outcome"] == "Fail").sum()
        wr = winrate(succ, fail)
        buckets.append(b); wrs.append(wr); counts.append(len(sub))

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(buckets, wrs)
    ax1.set_title("Win Rate by Signal Bucket")
    ax1.set_ylabel("Win Rate (%)")
    for i, v in enumerate(wrs):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(buckets, counts, marker="o")
    ax2.set_ylabel("Count")

    out = os.path.join(ANALYTICS_DIR, f"fig_winrate_by_bucket_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")

def plot_winrate_by_regime(df):
    stamp = ist_stamp()
    regimes = []
    wrs = []
    counts = []
    for regime, g in df.groupby(df["Regime"].fillna("Unknown")):
        succ = (g["Outcome"] == "Success").sum()
        fail = (g["Outcome"] == "Fail").sum()
        wr = winrate(succ, fail)
        regimes.append(regime); wrs.append(wr); counts.append(len(g))

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(regimes, wrs)
    ax1.set_title("Win Rate by Regime")
    ax1.set_ylabel("Win Rate (%)")
    for i, v in enumerate(wrs):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(regimes, counts, marker="o")
    ax2.set_ylabel("Count")

    out = os.path.join(ANALYTICS_DIR, f"fig_winrate_by_regime_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")

def plot_confidence_hist(df):
    stamp = ist_stamp()
    fig, ax = plt.subplots(figsize=(10,6))
    for b in BUCKET_ORDER:
        sub = pd.to_numeric(df.loc[df["Signal"] == b, "Confidence"], errors="coerce").dropna()
        if not sub.empty:
            ax.hist(sub, bins=20, alpha=0.5, label=b)
    ax.set_title("Confidence Distribution by Bucket")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.legend()
    out = os.path.join(ANALYTICS_DIR, f"fig_confidence_hist_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")

def plot_deserved_strong_wr(df):
    stamp = ist_stamp()
    # Build two groups
    groups = []
    wrs = []
    counts = []
    for flag, label in [("DeservedStrongBuy","Deserved Strong Buy"),
                        ("DeservedStrongSell","Deserved Strong Sell")]:
        if flag not in df.columns:
            continue
        sub = df[pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int) == 1]
        succ = (sub["Outcome"] == "Success").sum()
        fail = (sub["Outcome"] == "Fail").sum()
        wr = winrate(succ, fail)
        groups.append(label); wrs.append(wr); counts.append(len(sub))

    if not groups:
        print("[INFO] No DeservedStrong* columns found; skipping figure.")
        return

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.bar(groups, wrs)
    ax1.set_title("Win Rate of Deserved Strong Sets")
    ax1.set_ylabel("Win Rate (%)")
    for i, v in enumerate(wrs):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(groups, counts, marker="o")
    ax2.set_ylabel("Count")

    out = os.path.join(ANALYTICS_DIR, f"fig_deservedstrong_wr_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")

def plot_conf_threshold_sweep(df, side="Sell", min_conf=50, max_conf=90, step=5):
    """
    For side in {"Buy","Sell"} (includes Strong/regular), vary the confidence threshold and
    show resulting win rate & count if we promoted signals with Confidence>=thr.
    """
    stamp = ist_stamp()
    subset = df[df["Signal"].str.contains(side, na=False)].copy()
    if subset.empty:
        print(f"[INFO] No {side} signals; skipping sweep.")
        return
    thrs = []
    wrs = []
    counts = []
    for thr in range(min_conf, max_conf + 1, step):
        eligible = subset[pd.to_numeric(subset["Confidence"], errors="coerce").fillna(0) >= thr]
        succ = (eligible["Outcome"] == "Success").sum()
        fail = (eligible["Outcome"] == "Fail").sum()
        wr = winrate(succ, fail)
        thrs.append(thr); wrs.append(wr); counts.append(len(eligible))

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(thrs, wrs, marker="o")
    ax1.set_title(f"{side}: Win Rate vs Confidence Threshold")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Win Rate (%)")

    ax2 = ax1.twinx()
    ax2.plot(thrs, counts, marker="s")
    ax2.set_ylabel("Count >= Threshold")

    out = os.path.join(ANALYTICS_DIR, f"fig_conf_threshold_sweep_{side}_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")

def main():
    ensure_dir(ANALYTICS_DIR)
    df = load_all()

    # Basic figures
    plot_winrate_by_bucket(df)
    plot_winrate_by_regime(df)
    plot_confidence_hist(df)
    plot_deserved_strong_wr(df)

    # Threshold sweeps (optional but useful)
    plot_conf_threshold_sweep(df, side="Sell", min_conf=50, max_conf=90, step=5)
    plot_conf_threshold_sweep(df, side="Buy",  min_conf=50, max_conf=90, step=5)

if __name__ == "__main__":
    main()
