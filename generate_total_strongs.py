#!/usr/bin/env python3
"""
generate_total_strongs.py

Scan data_archive/*.json and generate a single CSV in analytics/ containing
only Strong signals (Strong Buy / Strong Sell). The CSV will contain
flattened top-level fields plus flattened analysis_log (components, indicator_scores,
confluence flags, hma_gatekeeper/supervisor info) and flattened indicators.

Usage:
    python generate_total_strongs.py

Output:
    analytics/TotalStrong_<IST-YYYY-MM-DD_HH-MM-SS>.csv
"""
import os
import json
import glob
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd

# CONFIG
DATA_ARCHIVE = "data_archive"
ANALYTICS_DIR = "analytics"
os.makedirs(ANALYTICS_DIR, exist_ok=True)

IST_TZ = pytz.timezone("Asia/Kolkata")

# Helpers
def utcnow_ist_str():
    now = datetime.now(timezone.utc).astimezone(IST_TZ)
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load JSON {path}: {e}")
        return None

def norm_key(k: str) -> str:
    return str(k).replace(" ", "_").replace("-", "_")

def flatten_indicators(indicators):
    """
    Flatten nested 'indicators' dict to IND__{key} or IND__{key}__{subkey}.
    Non-primitives (lists/dicts) are JSON-dumped.
    """
    flat = {}
    if not isinstance(indicators, dict):
        return flat
    for k, v in indicators.items():
        kk = norm_key(k)
        if isinstance(v, dict):
            for sk, sv in v.items():
                col = f"IND__{kk}__{norm_key(str(sk))}"
                if isinstance(sv, (dict, list)):
                    try:
                        flat[col] = json.dumps(sv)
                    except Exception:
                        flat[col] = str(sv)
                else:
                    flat[col] = sv
        else:
            col = f"IND__{kk}"
            if isinstance(v, (dict, list)):
                try:
                    flat[col] = json.dumps(v)
                except Exception:
                    flat[col] = str(v)
            else:
                flat[col] = v
    return flat

def flatten_analysis_log(analysis_log):
    """
    Flatten analysis_log into named columns:
      - components -> COMP__{name}
      - indicator_scores -> SCORE__{name}
      - confluence_flags -> CONFL__flags (semicolon-joined) and CONFL_FLAG__{flag} binary cols
      - would_be_strong_if -> WOULD_BE_STRONG (JSON string)
      - hma_gatekeeper -> HMA_GATEKEEPER__{k}
      - supervisor -> SUPERVISOR__{k}
      - keep ANALYSIS__engine and ANALYSIS__confidence
    """
    flat = {}
    if not isinstance(analysis_log, dict):
        return flat

    # components
    comps = analysis_log.get("components") or {}
    if isinstance(comps, dict):
        for k, v in comps.items():
            flat[f"COMP__{norm_key(k)}"] = v

    # indicator_scores
    ind_scores = analysis_log.get("indicator_scores") or {}
    if isinstance(ind_scores, dict):
        for k, v in ind_scores.items():
            key = f"SCORE__{norm_key(k)}"
            if isinstance(v, (dict, list)):
                try:
                    flat[key] = json.dumps(v)
                except Exception:
                    flat[key] = str(v)
            else:
                flat[key] = v

    # confluence_flags
    flags = analysis_log.get("confluence_flags") or []
    if isinstance(flags, (list, tuple)):
        flat["CONFL__flags"] = ";".join([str(x) for x in flags]) if flags else ""
        # individual binary columns
        for f in flags:
            col = f"CONFL_FLAG__{norm_key(str(f))}"
            flat[col] = 1
    else:
        flat["CONFL__flags"] = ""

    # would_be_strong_if (store JSON string)
    wbs = analysis_log.get("would_be_strong_if")
    try:
        flat["WOULD_BE_STRONG"] = json.dumps(wbs) if wbs is not None else ""
    except Exception:
        flat["WOULD_BE_STRONG"] = str(wbs)

    # hma_gatekeeper & supervisor
    hma = analysis_log.get("hma_gatekeeper")
    if isinstance(hma, dict):
        for k, v in hma.items():
            flat[f"HMA_GATEKEEPER__{norm_key(str(k))}"] = v
    else:
        # ensure columns exist (avoid missing columns later)
        flat.setdefault("HMA_GATEKEEPER__applied", None)
        flat.setdefault("HMA_GATEKEEPER__before_confidence", None)
        flat.setdefault("HMA_GATEKEEPER__after_confidence", None)
        flat.setdefault("HMA_GATEKEEPER__reason", None)

    sup = analysis_log.get("supervisor")
    if isinstance(sup, dict):
        for k, v in sup.items():
            flat[f"SUPERVISOR__{norm_key(str(k))}"] = v
    else:
        flat.setdefault("SUPERVISOR__applied", None)

    # engine & confidence
    flat["ANALYSIS__engine"] = analysis_log.get("engine")
    flat["ANALYSIS__confidence"] = analysis_log.get("confidence")

    return flat

def flatten_top_level(signal_obj):
    """
    Extract top-level well-known fields; for any other top-level keys, prefix with TOP__.
    """
    out = {}
    if not isinstance(signal_obj, dict):
        return out
    # common expected fields
    for k in ["coin", "price", "tp", "sl", "leverage", "confidence", "signal", "estimated_profit", "regime", "signal_time_utc", "signal_time_ist"]:
        if k in signal_obj:
            out[k.upper() if k.isupper() else k] = signal_obj.get(k)
    # keep any other top-level keys as TOP__{key}
    for k, v in signal_obj.items():
        if k in ["coin", "price", "tp", "sl", "leverage", "confidence", "signal", "estimated_profit", "regime", "signal_time_utc", "signal_time_ist", "analysis_log", "indicators"]:
            continue
        # add everything else (JSON-dump complex)
        key = f"TOP__{norm_key(str(k))}"
        if isinstance(v, (dict, list)):
            try:
                out[key] = json.dumps(v)
            except Exception:
                out[key] = str(v)
        else:
            out[key] = v
    return out

def process_json_file(path):
    """
    Return list of flattened dict rows (one per Strong signal) found in the file.
    """
    data = safe_load_json(path)
    if data is None:
        return []

    rows = []

    # JSON may be a list of signals or a single object containing 'signals' list
    candidates = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        # Some files might be { "signals": [...] } or single-signal dict
        if "signals" in data and isinstance(data["signals"], list):
            candidates = data["signals"]
        else:
            # single signal object
            candidates = [data]
    else:
        return []

    for sig in candidates:
        if not isinstance(sig, dict):
            continue
        sig_label = str(sig.get("signal") or "").strip().lower()
        if "strong" in sig_label:
            # keep this signal; flatten fields
            row = {}
            row["__source_file"] = os.path.basename(path)
            # top-level
            row.update(flatten_top_level(sig))
            # indicators
            row.update(flatten_indicators(sig.get("indicators") or {}))
            # analysis_log flattened
            row.update(flatten_analysis_log(sig.get("analysis_log") or {}))
            rows.append(row)
    return rows

def main():
    files = sorted(glob.glob(os.path.join(DATA_ARCHIVE, "*.json")))
    if not files:
        print("[INFO] No JSON files found in data_archive/. Nothing to do.")
        return

    all_rows = []
    scanned = 0
    strong_count = 0
    for f in files:
        scanned += 1
        rows = process_json_file(f)
        strong_count += len(rows)
        all_rows.extend(rows)

    if not all_rows:
        print(f"[INFO] Scanned {scanned} JSON files; found 0 Strong signals. No CSV produced.")
        return

    # create DataFrame
    df = pd.DataFrame(all_rows)

    # sort columns for readability: put source, coin, signal, confidence, price, tp, sl, regime, times first if present
    preferred = ["__source_file", "coin", "signal", "confidence", "price", "tp", "sl", "estimated_profit", "regime", "signal_time_utc", "signal_time_ist"]
    cols = preferred + [c for c in df.columns if c not in preferred]
    # keep only unique preserving order
    seen = set()
    cols_ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            cols_ordered.append(c)

    df = df[cols_ordered]

    # filename with IST timestamp (recreate fresh)
    fname = f"TotalStrong_{utcnow_ist_str()}.csv"
    outpath = os.path.join(ANALYTICS_DIR, fname)

    try:
        df.to_csv(outpath, index=False)
        print(f"[OK] Wrote {outpath} with {len(df)} strong rows (from {scanned} files).")
    except Exception as e:
        print(f"[ERROR] Could not write CSV {outpath}: {e}")
        return

if __name__ == "__main__":
    main()
