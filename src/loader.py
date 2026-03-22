import numpy as np
import pandas as pd
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


def load_file(file, filename: str = None) -> pd.DataFrame:
    """
    Load any supported file into a pandas DataFrame.

    Parameters:
        file     : file path (str) OR file-like object (from Streamlit uploader)
        filename : original filename — required when file is a file-like object

    Returns:
        df       : raw pandas DataFrame
    """

    # Get filename and extension
    if filename is None:
        filename = str(file)
    ext = os.path.splitext(filename)[1].lower()

    # ── Load based on format ──────────────────────────

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
                    temp = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content), sep=sep)
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
            f"Supported: .csv .xlsx .xls .json .tsv .txt .parquet .h5 .hdf5 .npy .npz"
        )

    return df


def detect_label_column(df: pd.DataFrame, hint: str = None):
    """
    Find the label column in a DataFrame.
    Returns (label_col_name, series) or (None, None) if not found.
    """
    # User-specified hint
    if hint and hint in df.columns:
        return hint, df[hint].astype(int)

    # Auto-detect by common names
    for name in LABEL_NAMES:
        if name in df.columns:
            return name, df[name].astype(int)

    return None, None


def preprocess(df: pd.DataFrame, is_timeseries: bool = False):
    """
    Preprocess a DataFrame for ML:
      - Optionally remove duplicates (skipped for time series)
      - Fill missing values
      - Encode categorical columns
      - Scale if needed

    Returns:
        X_scaled : preprocessed feature DataFrame
    """
    df = df.copy()

    # Remove duplicates only for non-timeseries data
    if not is_timeseries:
        df.drop_duplicates(inplace=True)

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)

    # Keep numeric only
    df = df.select_dtypes(include=[np.number])

    # Scale only if not already normalized
    if df.max().max() > 10 or df.min().min() < -1:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        df = pd.DataFrame(scaled, columns=df.columns)

    return df