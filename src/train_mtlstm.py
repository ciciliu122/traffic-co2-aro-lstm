import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model_mtlstm import MultiTaskLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "results/mtlstm_model.pth"
SCALER_PATH = "results/mtlstm_scaler.npy"

SEQ_LEN = 14  # 14-day input window
EPOCHS = 300
LR = 0.001
HIDDEN = 64
LAYERS = 2

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    # Normalize date
    df["date"] = pd.to_datetime(df["date"])

    # Pivot → wide format
    pivot = df.pivot_table(index="date", columns="sector", values="co2")
    pivot = pivot.sort_index()

    # Convert to numpy
    values = pivot.values.astype(float)

    # Standardize
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    # Save scaler
    np.save(SCALER_PATH, scaler.scale_)
    np.save("results/mtlstm_mean.npy", scaler.mean_)

    # Create sliding windows
    X, Y = [], []
    for i in range(len(values_scaled) - SEQ_LEN):
        X.append(values_scaled[i : i + SEQ_LEN])
        Y.append(values_scaled[i + SEQ_LEN])   # predict next day (7 outputs)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    return X, Y, scaler, pivot.columns.tolist()

def train():
    X, Y, scaler, sector_list = load_and_prepare()
    model = MultiTaskLSTM(
        input_dim=7, hidden_dim=HIDDEN, num_layers=LAYERS,
        output_dim=7, dropout=0.2
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training Multi-Task LSTM on {len(X)} samples...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        y_hat = model(X.to(DEVICE))
        loss = criterion(y_hat, Y.to(DEVICE))

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] Loss = {loss.item():.6f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    with open("results/mtlstm_meta.json", "w") as f:
        json.dump({"sectors": sector_list}, f)

    print("\n🎉 Training complete. Model saved!")

if __name__ == "__main__":
    train()
