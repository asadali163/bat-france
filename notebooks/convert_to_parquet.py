"""
Convert raw CSV files to optimised Parquet.
Run once locally:  python notebooks/convert_to_parquet.py
"""
import os
import pandas as pd

OUT_DIR = "./data/parquet"
os.makedirs(OUT_DIR, exist_ok=True)


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast low-cardinality object columns to category, floats to float32."""
    for col in df.select_dtypes("object").columns:
        # Skip date-like columns — keep them as plain strings so pd.to_datetime works cleanly on load
        if col.lower() == "date" or col.lower().endswith("_date"):
            continue
        n_unique = df[col].nunique()
        if n_unique / len(df) < 0.5:          # less than 50% unique → category
            df[col] = df[col].astype("category")
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    return df


# ── combined_df (sell data) ───────────────────────────────────────────────
print("Converting combined_df.csv …")
df = pd.read_csv("./data/cache/combined_df.csv", low_memory=False)
before = df.memory_usage(deep=True).sum() / 1e6
df = optimise_dtypes(df)
after = df.memory_usage(deep=True).sum() / 1e6
out = f"{OUT_DIR}/combined_df.parquet"
df.to_parquet(out, index=False)
size_mb = os.path.getsize(out) / 1e6
print(f"  Memory: {before:.0f} MB → {after:.0f} MB  ({before/after:.1f}x smaller)")
print(f"  File:   {size_mb:.1f} MB on disk\n")

# ── weather ───────────────────────────────────────────────────────────────
print("Converting weather_till_march2.csv …")
df_wx = pd.read_csv("./data/raw/weather_till_march2.csv", low_memory=False)
df_wx = optimise_dtypes(df_wx)
out_wx = f"{OUT_DIR}/weather_till_march2.parquet"
df_wx.to_parquet(out_wx, index=False)
print(f"  File: {os.path.getsize(out_wx)/1e6:.2f} MB\n")

# ── events ────────────────────────────────────────────────────────────────
print("Converting events_batch2.csv …")
df_ev = pd.read_csv("./data/raw/events_batch2.csv", low_memory=False)
df_ev = optimise_dtypes(df_ev)
out_ev = f"{OUT_DIR}/events_batch2.parquet"
df_ev.to_parquet(out_ev, index=False)
print(f"  File: {os.path.getsize(out_ev)/1e6:.2f} MB\n")

print("Done. Files written to", OUT_DIR)
