import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model_mtlstm import MultiTaskLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "results/mtlstm_model.pth"

def evaluate():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="sector", values="co2")
    pivot = pivot.sort_index()

    scaler = StandardScaler()
    scaler.scale_ = np.load("results/mtlstm_scaler.npy")
    scaler.mean_ = np.load("results/mtlstm_mean.npy")

    values = pivot.values.astype(float)
    values_scaled = scaler.transform(values)

    with open("results/mtlstm_meta.json", "r") as f:
        meta = json.load(f)
    sectors = meta["sectors"]

    # load model
    model = MultiTaskLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    seq_len = 14
    X = torch.tensor(values_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    preds_scaled = model(X).detach().cpu().numpy()[0]
    preds = preds_scaled * scaler.scale_ + scaler.mean_

    # Plot
    plt.figure(figsize=(14, 6))
    t = range(len(values))
    plt.plot(t, values[:, sectors.index("Total")], label="Real CO₂ (Total)", marker="o")
    plt.scatter(len(values), preds[sectors.index("Total")], color="r", label="Predicted Next Day")

    plt.title("Multi-Task LSTM CO₂ Forecasting (Total Sector)")
    plt.xlabel("Time Index")
    plt.ylabel("CO₂ Emission")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/multi_prediction_plot.png", dpi=200)

    print("\n Prediction figure saved → results/multi_prediction_plot.png")

if __name__ == "__main__":
    evaluate()
