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

def evaluate_full_sequence():
    print("Loading data...")

    # === Load & pivot ===
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="sector", values="co2")
    pivot = pivot.sort_index()

    # === Load scaler ===
    scaler = StandardScaler()
    scaler.scale_ = np.load("results/mtlstm_scaler.npy")
    scaler.mean_ = np.load("results/mtlstm_mean.npy")

    values = pivot.values.astype(float)
    values_scaled = scaler.transform(values)

    total_real = values[:, list(pivot.columns).index("Total")]

    # === meta ===
    with open("results/mtlstm_meta.json", "r") as f:
        meta = json.load(f)
    sectors = meta["sectors"]

    # === load model ===
    model = MultiTaskLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    seq_len = 14
    n = len(values)

    # ============ FULL-SEQUENCE ROLLING PREDICTION ============
    preds = []

    # 初始滑动窗口
    window = values_scaled[:seq_len].copy()

    for i in range(seq_len, n):
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pred_scaled = model(x).detach().cpu().numpy()[0]
        pred = pred_scaled * scaler.scale_ + scaler.mean_

        preds.append(pred[sectors.index("Total")])

        # 更新滑动窗口（真实滚动 or 预测滚动都可，这里用真实滚动）
        new_row = values_scaled[i]
        window = np.vstack([window[1:], new_row])

    preds = np.array(preds)

    # 同真实数据对齐
    real = total_real[seq_len:]

    # ===================== PLOT ===============================
    plt.figure(figsize=(14, 6))
    plt.plot(real, label="Real CO₂ (Total)", marker="o", markersize=3)
    plt.plot(preds, label="Predicted CO₂ (Total)", marker="x", markersize=3, color="orange")

    plt.title("Multi-Task LSTM — Full Sequence CO₂ Forecast vs Real (Total Sector)")
    plt.xlabel("Time Index")
    plt.ylabel("CO₂ Emission")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig("results/full_sequence_vs_real.png", dpi=200)

    print("\n Full-sequence comparison saved → results/full_sequence_vs_real.png")


if __name__ == "__main__":
    evaluate_full_sequence()
