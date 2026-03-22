import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# PATHS — all 3 MSL files
# ─────────────────────────────────────────────
TRAIN_PATH = "data/MSL_train.npy"
TEST_PATH  = "data/MSL_test.npy"
LABEL_PATH = "data/MSL_label.npy"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — Load all 3 files
# ─────────────────────────────────────────────
print("Loading MSL dataset files...")

train_data = np.load(TRAIN_PATH, allow_pickle=True)
test_data  = np.load(TEST_PATH,  allow_pickle=True)
labels     = np.load(LABEL_PATH, allow_pickle=True)

print(f"Train shape  : {train_data.shape}")
print(f"Test shape   : {test_data.shape}")
print(f"Labels shape : {labels.shape}")

# ─────────────────────────────────────────────
# STEP 2 — Create column names
# ─────────────────────────────────────────────
n_sensors = train_data.shape[1]
columns = [f"sensor_{i+1:02d}" for i in range(n_sensors)]

# ─────────────────────────────────────────────
# STEP 3 — Convert to DataFrames
# NOTE: No duplicate removal — this is time
# series data, repeated readings are valid
# ─────────────────────────────────────────────
train_df = pd.DataFrame(train_data, columns=columns)
test_df  = pd.DataFrame(test_data,  columns=columns)

# Attach labels (False=0 normal, True=1 anomaly)
test_df["faulty"] = labels.astype(int)

# ─────────────────────────────────────────────
# STEP 4 — Print summary
# ─────────────────────────────────────────────
print("\n--- Train Data ---")
print(f"Rows    : {len(train_df)}")
print(f"Columns : {len(train_df.columns)}")

print("\n--- Test Data ---")
print(f"Rows    : {len(test_df)}")
print(f"Normal  (0) : {(test_df['faulty'] == 0).sum()}")
print(f"Anomaly (1) : {(test_df['faulty'] == 1).sum()}")

# ─────────────────────────────────────────────
# STEP 5 — Save both files
# ─────────────────────────────────────────────
train_df.to_csv(f"{OUTPUT_DIR}/cleaned_train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/cleaned_test.csv",   index=False)

print(f"\n✅ Saved cleaned_train.csv → {OUTPUT_DIR}/")
print(f"✅ Saved cleaned_test.csv  → {OUTPUT_DIR}/")