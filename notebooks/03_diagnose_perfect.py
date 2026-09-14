"""Diagnose suspicious 100% accuracy: is the target trivially predictable?"""
import sys
from pathlib import Path

# Add project root to sys.path so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.config import load_config
from src.data.load import load_raw
from src.features.build import add_features

cfg = load_config()
df = load_raw(cfg)
df = add_features(df, date_column=cfg["data"].get("date_column"))

target = cfg["data"]["target"]
drop = cfg["data"]["drop_columns"]
X = df.drop(columns=[target] + [c for c in drop if c in df.columns])
y = LabelEncoder().fit_transform(df[target])

print("=" * 70)
print("Single-feature decision stump (depth=1) on each numeric column")
print("=" * 70)
for col in X.select_dtypes(include="number").columns:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=1, random_state=42),
        X[[col]], y, cv=3, scoring="f1"
    )
    m = scores.mean()
    flag = "  <-- SINGLE FEATURE PREDICTS TARGET" if m > 0.9 else ""
    print(f"  {col:<35} F1 = {m:.4f}{flag}")

print()
print("=" * 70)
print("Range overlap check: does a single threshold separate the classes?")
print("=" * 70)
pos = df[df[target] == "Yes"]
neg = df[df[target] == "No"]

for col in X.select_dtypes(include="number").columns:
    pos_min = pos[col].min()
    pos_max = pos[col].max()
    neg_min = neg[col].min()
    neg_max = neg[col].max()
    if pos_min > neg_max or pos_max < neg_min:
        print(f"  {col:<35} PERFECT SEPARATION")
        print(f"      pos range: [{pos_min}, {pos_max}]")
        print(f"      neg range: [{neg_min}, {neg_max}]")

print()
print("=" * 70)
print("Categorical features: distribution by target class")
print("=" * 70)
for col in X.select_dtypes(include="object").columns:
    print(f"\n{col}")
    print(pd.crosstab(df[col], df[target], normalize="index").round(3))