import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ══════════════════════════════════════════════════════
#  SUPPORTED FORMATS
#  .csv  .xlsx  .xls  .json  .tsv  .txt
#  .npy  .npz   .parquet  .h5  .hdf5
# ══════════════════════════════════════════════════════

# Common label column names — auto-detected
LABEL_NAMES = [
    "faulty", "label", "labels", "anomaly", "target",
    "class", "fault", "failure", "defect", "is_anomaly",
    "is_fault", "status", "y"
]

# ══════════════════════════════════════════════════════
#  KNOWN DATASET SENSOR MAPS
#  Add new dataset entries here as the project grows.
#  Key format: exact column count → list of names
# ══════════════════════════════════════════════════════

# NASA MSL (Mars Science Lab) — 55 telemetry channels
MSL_SENSOR_NAMES = [
    "radiation_flux",       # 01
    "solar_wind_speed",     # 02
    "magnetic_field_x",     # 03
    "magnetic_field_y",     # 04
    "magnetic_field_z",     # 05
    "temperature_1",        # 06
    "temperature_2",        # 07
    "pressure_1",           # 08
    "pressure_2",           # 09
    "humidity",             # 10
    "vibration_x",          # 11
    "vibration_y",          # 12
    "vibration_z",          # 13
    "current_draw",         # 14
    "voltage_1",            # 15
    "voltage_2",            # 16
    "torque",               # 17
    "rpm",                  # 18
    "flow_rate",            # 19
    "accel_x",              # 20
    "accel_y",              # 21
    "accel_z",              # 22
    "gyro_x",               # 23
    "gyro_y",               # 24
    "gyro_z",               # 25
    "power_output",         # 26
    "thermal_load",         # 27
    "optical_density",      # 28
    "spectrometer",         # 29
    "altimeter",            # 30
    "barometric_pressure",  # 31
    "co2_level",            # 32
    "o2_level",             # 33
    "methane_ppm",          # 34
    "dust_particle",        # 35
    "uv_index",             # 36
    "ir_flux",              # 37
    "relay_state",          # 38
    "valve_position",       # 39
    "pump_speed",           # 40
    "motor_load",           # 41
    "battery_soc",          # 42
    "charge_rate",          # 43
    "discharge_rate",       # 44
    "coolant_temp",         # 45
    "lubricant_level",      # 46
    "encoder_pos",          # 47
    "signal_strength",      # 48
    "error_flag",           # 49
    "status_code",          # 50
    "latency_ms",           # 51
    "packet_loss",          # 52
    "memory_usage",         # 53
    "cpu_temp",             # 54
    "watchdog_timer",       # 55
]

# NASA SMAP (Soil Moisture Active Passive) — 25 channels
SMAP_SENSOR_NAMES = [
    "radiation_flux",
    "solar_panel_v",
    "battery_charge",
    "temperature_1",
    "temperature_2",
    "pressure",
    "humidity",
    "vibration",
    "current_draw",
    "voltage",
    "power_output",
    "thermal_load",
    "motor_speed",
    "flow_rate",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "magnetic_x",
    "magnetic_y",
    "magnetic_z",
    "co2_ppm",
    "o2_level",
]

# Lookup: (col_count, filename_keywords) → name list
DATASET_REGISTRY = [
    (55, ["msl"],  MSL_SENSOR_NAMES),
    (25, ["smap"], SMAP_SENSOR_NAMES),
]

# Generic industrial fallback names (used when dataset is unknown)
_GENERIC_NAMES = [
    "temperature", "pressure", "vibration", "humidity",
    "flow_rate", "voltage", "current", "rpm", "torque",
    "power", "temperature_2", "pressure_2", "vibration_2",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y",
    "gyro_z", "magnetic_x", "magnetic_y", "magnetic_z",
    "co2_ppm", "o2_level", "noise_db", "load_pct",
    "efficiency", "heat_flux", "coolant_temp", "oil_pressure",
    "battery_v", "signal_strength", "error_count",
    "status", "latency_ms",
]


# ══════════════════════════════════════════════════════
#  AUTO SENSOR NAME DETECTION
# ══════════════════════════════════════════════════════

def auto_detect_sensor_names(df: pd.DataFrame, filename: str = "") -> list:
    """
    Automatically assign meaningful sensor names to DataFrame columns.

    Priority order:
    1. Columns already have real names (not sensor_XX / integer) → keep as-is
    2. Filename or column count matches a known dataset → use its names
    3. Statistical heuristics (value range, variance, uniqueness) → guess type
    4. Generic industrial names (temperature, pressure, vibration…)
    5. Last resort: sensor_01, sensor_02, …

    Parameters:
        df       : input DataFrame (before or after preprocessing)
        filename : original file name — improves dataset detection

    Returns:
        list[str] : column names, same length as df.columns
    """
    cols = list(df.columns)
    n    = len(cols)

    # ── 1. Already real names? ───────────────────────────
    all_generic = all(
        str(c).startswith("sensor_") or str(c).isdigit() or str(c) == str(i)
        for i, c in enumerate(cols)
    )
    if not all_generic:
        return cols   # keep existing names

    # ── 2. Match known dataset ───────────────────────────
    fname = filename.lower()
    for expected_n, keywords, name_list in DATASET_REGISTRY:
        name_match = any(kw in fname for kw in keywords)
        size_match = n == expected_n
        if name_match or size_match:
            names = name_list[:n]
            # Pad with generic sensor_XX if fewer names than columns
            if len(names) < n:
                names = names + [f"sensor_{i+1:02d}" for i in range(len(names), n)]
            return names

    # ── 3. Statistical heuristic per column ─────────────
    numeric_df = df.select_dtypes(include=[np.number])
    assigned   = []
    used_names = set()

    for col in cols:
        if col not in numeric_df.columns:
            assigned.append(str(col))
            continue

        series   = numeric_df[col].dropna()
        if len(series) == 0:
            assigned.append(f"sensor_{len(assigned)+1:02d}")
            continue

        col_min  = float(series.min())
        col_max  = float(series.max())
        col_mean = float(series.mean())
        col_std  = float(series.std()) if len(series) > 1 else 0.0
        n_unique = series.nunique()
        value_range = col_max - col_min

        # Detect if data is already normalized (StandardScaler output)
        is_normalized = (abs(col_mean) < 1.0 and 0.5 < col_std < 3.0
                         and col_min > -10 and col_max < 10)

        # Rule-based assignment
        if n_unique <= 2:
            name = "fault_flag"
        elif is_normalized:
            # Normalized data: use column position index as differentiator
            # (values are all similar ranges after StandardScaler)
            # Fall through to generic industrial names below
            name = None
        elif 0 <= col_min and col_max >= 500:
            name = "rpm"
        elif -50 <= col_min and col_max <= 150 and col_std < 40:
            name = "temperature"
        elif 0 <= col_min and col_max <= 15 and col_std < 4:
            name = "pressure"
        elif 0 <= col_min and col_max <= 100 and 20 < col_mean < 80:
            name = "humidity"
        elif -5 <= col_min and col_max <= 5 and col_std < 2:
            name = "vibration"
        elif -2 <= col_min and col_max <= 2 and col_std < 1:
            name = "accel"
        elif 0 <= col_min and col_max <= 1.05 and col_std < 0.4:
            name = "normalized_signal"
        elif col_min >= 0 and col_max <= 30 and col_mean < 15:
            name = "flow_rate"
        elif col_min >= 0 and col_max <= 500 and col_std < 100:
            name = "voltage"
        elif col_min >= 0 and col_max <= 50 and col_std < 20:
            name = "current"
        else:
            name = "sensor_signal"

        # For normalized data: use generic industrial names by position
        if name is None:
            idx = len(assigned)
            name = _GENERIC_NAMES[idx] if idx < len(_GENERIC_NAMES) else f"sensor_{idx+1:02d}"

        # Deduplicate: temperature → temperature_2 → temperature_3 …
        if name in used_names:
            idx = 2
            while f"{name}_{idx}" in used_names:
                idx += 1
            name = f"{name}_{idx}"

        used_names.add(name)
        assigned.append(name)

    return assigned


def apply_sensor_names(df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
    """
    Convenience wrapper — returns a copy of df with sensor names applied.

    Usage in dashboard.py:
        train_raw = apply_sensor_names(load_file(f, f.name), f.name)
    """
    out = df.copy()
    out.columns = auto_detect_sensor_names(df, filename)
    return out


# ══════════════════════════════════════════════════════
#  FILE LOADER
# ══════════════════════════════════════════════════════

def load_file(file, filename: str = None) -> pd.DataFrame:
    """
    Load any supported file into a pandas DataFrame.

    Parameters:
        file     : file path (str) OR file-like object (from Streamlit uploader)
        filename : original filename — required when file is a file-like object

    Returns:
        df       : raw pandas DataFrame
    """
    if filename is None:
        filename = str(file)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)

    elif ext == ".json":
        df = pd.read_json(file)

    elif ext == ".tsv":
        df = pd.read_csv(file, sep="\t")

    elif ext == ".txt":
        df = None
        for sep in [",", "\t", " ", ";"]:
            try:
                import io
                if hasattr(file, "read"):
                    content = file.read()
                    file.seek(0)
                    temp = pd.read_csv(
                        io.BytesIO(content) if isinstance(content, bytes)
                        else io.StringIO(content), sep=sep
                    )
                else:
                    temp = pd.read_csv(file, sep=sep)
                if temp.shape[1] > 1:
                    df = temp
                    break
            except Exception:
                continue
        if df is None:
            raise ValueError("Could not parse .txt file. Try saving it as .csv.")

    elif ext == ".parquet":
        df = pd.read_parquet(file)

    elif ext in [".h5", ".hdf5"]:
        import h5py
        if hasattr(file, "read"):
            import io, tempfile
            content = file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            with h5py.File(tmp_path, "r") as f:
                key = list(f.keys())[0]
            df = pd.read_hdf(tmp_path, key=key)
            os.unlink(tmp_path)
        else:
            with h5py.File(file, "r") as f:
                key = list(f.keys())[0]
            df = pd.read_hdf(file, key=key)

    elif ext == ".npy":
        if hasattr(file, "read"):
            data = np.load(file, allow_pickle=True)
        else:
            data = np.load(file, allow_pickle=True)

        if data.dtype.names:
            df = pd.DataFrame(data)
        elif data.dtype == object:
            try:
                df = pd.DataFrame(data.tolist())
            except Exception:
                df = pd.DataFrame(data)
        elif data.ndim == 1:
            df = pd.DataFrame(data, columns=["sensor_01"])
        elif data.ndim == 2:
            cols = [f"sensor_{i+1:02d}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=cols)
        else:
            raise ValueError(f"Unsupported .npy shape: {data.shape}")

    elif ext == ".npz":
        if hasattr(file, "read"):
            import io
            data_bytes = file.read()
            archive = np.load(io.BytesIO(data_bytes), allow_pickle=True)
        else:
            archive = np.load(file, allow_pickle=True)

        key = list(archive.keys())[0]
        data = archive[key]

        if data.ndim == 1:
            df = pd.DataFrame(data, columns=["sensor_01"])
        elif data.ndim == 2:
            cols = [f"sensor_{i+1:02d}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=cols)
        else:
            raise ValueError(f"Unsupported .npz shape: {data.shape}")

    else:
        raise ValueError(
            f"Unsupported format: '{ext}'\n"
            "Supported: .csv .xlsx .xls .json .tsv .txt .parquet .h5 .hdf5 .npy .npz"
        )

    return df


# ══════════════════════════════════════════════════════
#  LABEL DETECTION
# ══════════════════════════════════════════════════════

def detect_label_column(df: pd.DataFrame, hint: str = None):
    """
    Find the label column in a DataFrame.
    Returns (label_col_name, series) or (None, None) if not found.
    """
    if hint and hint in df.columns:
        return hint, df[hint].astype(int)

    for name in LABEL_NAMES:
        if name in df.columns:
            return name, df[name].astype(int)

    return None, None


# ══════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, is_timeseries: bool = False):
    """
    Preprocess a DataFrame for ML:
      - Remove duplicates (skipped for time series)
      - Fill missing values
      - Encode categorical columns
      - Scale if needed

    Returns:
        X_scaled : preprocessed feature DataFrame
    """
    df = df.copy()

    if not is_timeseries:
        df.drop_duplicates(inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)

    df = df.select_dtypes(include=[np.number])

    if df.max().max() > 10 or df.min().min() < -1:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)

    return df