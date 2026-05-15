# explore.py
# PURPOSE: Understand the UCI Heart Disease dataset before touching anything.

import pandas as pd

# ── LOAD DATASET ──────────────────────────────────────────────
df = pd.read_csv('data/heart_disease_uci.csv')

# ── DATASET SIZE ──────────────────────────────────────────────
print("=" * 60)
print("DATASET SIZE:")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")

# ── COLUMN NAMES AND TYPES ────────────────────────────────────
print("\n" + "=" * 60)
print("COLUMNS AND TYPES:")
print(df.dtypes)

# ── FIRST 5 ROWS ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("FIRST 5 ROWS:")
print(df.head())

# ── MISSING VALUES ────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# ── TARGET DISTRIBUTION ───────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET (num) DISTRIBUTION — raw 0–4 scale:")
print(df['num'].value_counts().sort_index())
print("\nAFTER BINARISING  (0 = No Disease  |  1 = Disease):")
binary = (df['num'] > 0).astype(int)
print(binary.value_counts())

# ── CATEGORICAL FEATURE VALUES ────────────────────────────────
print("\n" + "=" * 60)
print("UNIQUE VALUES FOR CATEGORICAL COLUMNS:")
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
    print(f"  {col:10s} → {df[col].unique().tolist()}")

# ── NUMERICAL STATISTICS ──────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICS FOR NUMERICAL COLUMNS:")
print(df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak']].describe())

print("\n✅  Exploration complete.")
