import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_loader import prepare_data
from model_lstm import MultiTaskLSTM
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model():

    # ---- Load data ----
    X_train, y_train, X_test, y_test, scaler, raw_df = prepare_data("data/sample_data.csv")

    # ---- Load hyperparameters ----
    if not os.path.exists("results/best_params.json"):
        raise FileNotFoundError("Missing best_params.json. Run train.py first!")

    with open("results/best_params.json", "r") as f:
        hparams = json.load(f)

    hidden_dim = hparams["hidden_dim"]
    num_layers = hparams["num_layers"]
    lr = hparams["learning_rate"]

    print(f"Loaded best hyperparameters: {hparams}")

    # ---- Load model with correct architecture ----
    model = MultiTaskLSTM(
        input_dim=7,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=7
    ).to(DEVICE)

    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    # ---- Convert test set ----
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    # ---- Inverse scaling ----
    inv_preds = scaler.inverse_transform(preds)
    inv_real = scaler.inverse_transform(y_test)

    # ---- Plot Total Sector ----
    real_total = inv_real[:, 6]
    pred_total = inv_preds[:, 6]

    plt.figure(figsize=(10, 4))
    plt.plot(real_total, label="Real CO2 (Total)", marker="o")
    plt.plot(pred_total, label="Predicted CO2 (Total)", marker="x")
    plt.title("CO₂ Forecasting (Total Sector)")
    plt.xlabel("Time Step")
    plt.ylabel("CO₂ Emission")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/prediction_plot.png", dpi=300)
    plt.close()

    print("Prediction plot saved → results/prediction_plot.png")


if __name__ == "__main__":
    evaluate_model()
