import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_loader import prepare_data
from model_lstm import MultiTaskLSTM
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model_path="results/best_model.pth"):
    """
    Load trained model -> predict -> inverse scale -> plot.
    """
    # ---- 1. Load data ----
    X_train, y_train, X_test, y_test, scaler, raw_df = prepare_data("data/sample_data.csv")

    # ---- 2. Load model ----
    model = MultiTaskLSTM()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Convert test set to torch
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    # ---- 3. Prediction ----
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    # ---- 4. Inverse scaling ----
    # scaler fitted on 7-sector wide format
    inv_preds = scaler.inverse_transform(preds)
    inv_real = scaler.inverse_transform(y_test)

    # ---- 5. Plot prediction for "Total" sector ----
    # "Total" is last column → index 6
    real_total = inv_real[:, 6]
    pred_total = inv_preds[:, 6]

    plt.figure(figsize=(10, 4))
    plt.plot(real_total, label="Real CO₂ (Total)", marker="o")
    plt.plot(pred_total, label="Predicted CO₂ (Total)", marker="x")
    plt.title("CO₂ Emission Forecast (Total Sector)")
    plt.xlabel("Time Step")
    plt.ylabel("CO₂ Emission")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/prediction_plot.png", dpi=300)
    plt.close()

    print("✅ Prediction plot saved → results/prediction_plot.png")


if __name__ == "__main__":
    evaluate_model()
