"""
Run this FIRST to discover the exact schema:
    python notebooks/01_inspect.py
"""
import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/raw/social_media_user_behavior.csv")

if not CSV_PATH.exists():
    raise SystemExit(f"[!] CSV not found at {CSV_PATH.resolve()}")

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
print("=" * 70)

print("\n--- Columns and dtypes ---")
for c in df.columns:
    print(f"  {c:<35} {str(df[c].dtype):<10}  nunique={df[c].nunique()}")

print("\n--- First 3 rows ---")
print(df.head(3).T)

print("\n--- Null counts ---")
print(df.isna().sum().sort_values(ascending=False))

print("\n--- Numeric summary ---")
print(df.describe().T)

print("\n--- Categorical values (first 10 of each) ---")
for c in df.select_dtypes(include=["object", "category"]).columns:
    vals = df[c].unique()[:10]
    print(f"  {c}: {list(vals)}")