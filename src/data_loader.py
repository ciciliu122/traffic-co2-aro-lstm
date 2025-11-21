import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SECTORS = [
    "Domestic Aviation",
    "Ground Transport",
    "Industry",
    "International Aviation",
    "Power",
    "Residential",
    "Total"
]

def load_and_pivot(csv_path):
    """
    Load long-format CO2 data and pivot to wide-format:
    date | 7 sectors
    """
    df = pd.read_csv(csv_path)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Pivot: each row = one day, columns = sectors
    pivot = df.pivot(index="date", columns="sector", values="co2")

    # Sort by date
    pivot = pivot.sort_index()

    # Ensure all sectors exist
    for s in SECTORS:
        if s not in pivot.columns:
            pivot[s] = np.nan

    # Forward fill missing values (demo dataset safety)
    pivot = pivot.fillna(method="ffill").fillna(method="bfill")

    return pivot


def create_sequences(data, seq_len=5):
    """
    Convert wide-format data → supervised learning sequences.
    X: (samples, seq_len, 7)
    y: (samples, 7)
    """
    X, y = [], []
    values = data.values  # shape: (days, 7)

    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])

    return np.array(X), np.array(y)


def prepare_data(csv_path, seq_len=5):
    """
    Full pipeline:
    - load & pivot
    - scale
    - sequence → train/test split
    """
    data = load_and_pivot(csv_path)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = create_sequences(pd.DataFrame(scaled, index=data.index), seq_len)

    # 80/20 split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test, scaler, data
