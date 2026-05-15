# preprocess.py
# PURPOSE: Clean the UCI Heart Disease dataset and save ready-to-train arrays.
#
# FEATURES USED (10 clinical measurements):
#   age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak
#
# DROPPED (too many nulls – would be mostly imputed noise):
#   ca (66 % null), thal (53 % null), slope (34 % null), id, dataset
#
# TARGET:  num > 0  →  1 (Heart Disease)
#           num == 0 →  0 (No Disease)
#
# TASK:    Binary Classification

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
df = pd.read_csv('data/heart_disease_uci.csv')
print(f"  Original shape : {df.shape}")

# ── STEP 1: DROP IRRELEVANT / HIGH-NULL COLUMNS ───────────────
df = df.drop(columns=['id', 'dataset', 'slope', 'ca', 'thal'])
print(f"  After column drop : {df.shape}")

# ── STEP 2: CREATE BINARY TARGET ─────────────────────────────
# num = 0  → no heart disease  (label 0)
# num > 0  → heart disease     (label 1)
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns=['num'])
print("\nTarget distribution:")
print(df['target'].value_counts())

# ── STEP 3: ENCODE CATEGORICAL FEATURES ──────────────────────
# sex: Male → 1, Female → 0
df['sex'] = (df['sex'] == 'Male').astype(int)

# fbs / exang: TRUE → 1, FALSE → 0
df['fbs']   = df['fbs'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})

# cp (chest pain type) → one-hot encode
cp_dummies = pd.get_dummies(df['cp'], prefix='cp')
df = pd.concat([df.drop(columns=['cp']), cp_dummies], axis=1)

# restecg → one-hot encode
restecg_dummies = pd.get_dummies(df['restecg'], prefix='restecg')
df = pd.concat([df.drop(columns=['restecg']), restecg_dummies], axis=1)

print(f"\n  After encoding, shape : {df.shape}")
print("  Final columns :", df.columns.tolist())

# ── STEP 4: HANDLE REMAINING MISSING VALUES ───────────────────
# Numerical: fill with median
for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: filled {df[col].isnull().sum()} remaining nulls with median {median_val:.1f}")

# Binary/encoded: fill with mode
for col in ['fbs', 'exang']:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

print(f"\n  Nulls after imputation: {df.isnull().sum().sum()}")
df['chol_per_age'] = df['chol'] / df['age']
df['bp_ratio'] = df['trestbps'] / df['age']

# ── STEP 5: SPLIT FEATURES AND TARGET ────────────────────────
X = df.drop(columns=['target']).values.astype(np.float32)
y = df['target'].values.astype(np.int64)
print(f"\n  X shape : {X.shape}")
print(f"  y shape : {y.shape}")

# Save feature names for the web app
feature_names = df.drop(columns=['target']).columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"\n  Feature names saved: {feature_names}")

# ── STEP 6: TRAIN / TEST SPLIT ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ── STEP 7: STANDARDISE NUMERICAL FEATURES ───────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── STEP 8: SAVE EVERYTHING ───────────────────────────────────
np.save('X_train.npy', X_train.astype(np.float32))
np.save('X_test.npy',  X_test.astype(np.float32))
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅  Preprocessing complete! Saved files:")
print("    X_train.npy  X_test.npy  y_train.npy  y_test.npy")
print("    scaler.pkl   feature_names.pkl")
