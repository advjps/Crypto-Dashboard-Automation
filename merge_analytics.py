# merge_analytics.py
import os
import glob
from datetime import datetime
import pytz
import pandas as pd

ANALYTICS_DIR = "analytics"
os.makedirs(ANALYTICS_DIR, exist_ok=True)   # <-- add this line

ALL_SIGNALS_CSV = os.path.join(ANALYTICS_DIR, "all_signals.csv")

def ist_timestamp():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d_%H-%M-%S")

def load_all_csvs():
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "signals_*.csv")))
    if not files:
        print(f"[INFO] No per-file analytics CSVs found in {ANALYTICS_DIR}/")
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def winrate(success, fail):
    denom = (success or 0) + (fail or 0)
    return (success / denom * 100.0) if denom > 0 else 0.0

def summarize_bucket(df, bucket_name):
    sub = df[df["signal"] == bucket_name].copy()
    if sub.empty:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    success = (sub["Outcome"] == "Success").sum()
    fail = (sub["Outcome"] == "Fail").sum()
    inconcl = (sub["Outcome"] == "Inconclusive").sum()
    wr = winrate(success, fail)
    avg_conf = pd.to_numeric(sub["Confidence"], errors="coerce").dropna().mean() or 0.0
    return {"Total": int(len(sub)), "Success": int(success), "Fail": int(fail),
            "Inconclusive": int(inconcl), "WinRate": wr, "AvgConfidence": float(avg_conf)}

def regime_summary(df):
    if df.empty or "Regime" not in df.columns:
        return pd.DataFrame()
    rows = []
    for regime, g in df.groupby("Regime"):
        if regime in (None, "", float("nan")):
            regime = "Unknown"
        succ = (g["Outcome"] == "Success").sum()
        fail = (g["Outcome"] == "Fail").sum()
        inc = (g["Outcome"] == "Inconclusive").sum()
        wr = winrate(succ, fail)
        avg_conf = pd.to_numeric(g["Confidence"], errors="coerce").dropna().mean() or 0.0
        rows.append({
            "Regime": regime, "Total": len(g), "Success": int(succ), "Fail": int(fail),
            "Inconclusive": int(inc), "WinRate": wr, "AvgConfidence": avg_conf
        })
    return pd.DataFrame(rows).sort_values(["Regime"])

def deserved_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    for flag_col, label in [("DeservedStrongBuy", "DeservedStrongBuy"),
                            ("DeservedStrongSell", "DeservedStrongSell")]:
        if flag_col not in df.columns:
            continue
        sub = df[pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        succ = (sub["Outcome"] == "Success").sum()
        fail = (sub["Outcome"] == "Fail").sum()
        inc  = (sub["Outcome"] == "Inconclusive").sum()
        wr = winrate(succ, fail)
        avg_conf = pd.to_numeric(sub["Confidence"], errors="coerce").dropna().mean() or 0.0
        out.append({
            "Group": label, "Total": len(sub), "Success": int(succ), "Fail": int(fail),
            "Inconclusive": int(inc), "WinRate": wr, "AvgConfidence": avg_conf
        })
    return pd.DataFrame(out)

def main():
    df = load_all_csvs()
    if df.empty:
        print("[INFO] Nothing to merge. Exiting.")
        return

    # Save stacked all_signals.csv
    df.to_csv(ALL_SIGNALS_CSV, index=False)
    print(f"[OK] Wrote {ALL_SIGNALS_CSV} with {len(df)} rows.")

    # --- Global bucket summary (by Signal) ---
    buckets = ["Strong Buy", "Buy", "Strong Sell", "Sell"]
    rows = []
    for b in buckets:
        rows.append({"Section": b, **summarize_bucket(df, b)})
    bucket_df = pd.DataFrame(rows)

    # --- Regime summary ---
    regime_df = regime_summary(df)

    # --- DeservedStrong summaries (analysis-only) ---
    deserved_df = deserved_summary(df)

    # Save CSV summaries
    stamp = ist_timestamp()
    bucket_csv = os.path.join(ANALYTICS_DIR, f"summary_by_signal_{stamp}.csv")
    regime_csv = os.path.join(ANALYTICS_DIR, f"summary_by_regime_{stamp}.csv")
    deserved_csv = os.path.join(ANALYTICS_DIR, f"summary_deserved_{stamp}.csv")
    bucket_df.to_csv(bucket_csv, index=False)
    regime_df.to_csv(regime_csv, index=False)
    deserved_df.to_csv(deserved_csv, index=False)

    # Human-readable summary.txt
    summary_txt = os.path.join(ANALYTICS_DIR, f"summary_{stamp}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("============== GLOBAL SUMMARY ==============\n\n")
        f.write("By Signal Bucket\n")
        f.write("-------------------------------------------\n")
        if not bucket_df.empty:
            f.write(bucket_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("By Regime\n")
        f.write("-------------------------------------------\n")
        if not regime_df.empty:
            f.write(regime_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("Deserved Strong (Analysis-only)\n")
        f.write("-------------------------------------------\n")
        if not deserved_df.empty:
            f.write(deserved_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n")
        else:
            f.write("(no data)\n")

    print(f"[OK] Wrote summaries:\n - {bucket_csv}\n - {regime_csv}\n - {deserved_csv}\n - {summary_txt}")

if __name__ == "__main__":
    main()
