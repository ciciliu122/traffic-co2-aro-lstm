import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import prepare_data
from model_lstm import MultiTaskLSTM
from aro_optimizer import AROOptimizer
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================
# Training function (returned loss)
# ===========================
def train_one_setting(hparams, X_train, y_train, X_val, y_val):
    """
    Train LSTM with specific hyperparameters and return validation loss.
    Used by ARO during search.
    """
    hidden_dim = hparams["hidden_dim"]
    num_layers = hparams["num_layers"]
    lr = hparams["learning_rate"]

    model = MultiTaskLSTM(
        input_dim=7,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=7
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert numpy → torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    # short training for ARO (fast)
    for epoch in range(20):  
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()

    return val_loss


# ===========================
# Wrapper for ARO (needed by optimizer)
# ===========================
def fitness_function(hparams):
    """Call train_one_setting using global cached data."""
    return train_one_setting(
        hparams,
        GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN,
        GLOBAL_X_VAL, GLOBAL_Y_VAL
    )


# ===========================
# Final full training using best hyperparameters
# ===========================
def train_final_model(hparams, X_train, y_train):
    model = MultiTaskLSTM(
        input_dim=7,
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        output_dim=7
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    # longer training for final model
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"[Final Model] Epoch {epoch+1}/80 - Loss: {loss.item():.6f}")

    # Save
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/best_model.pth")
    print("Best model saved → results/best_model.pth")

    return model


# ===========================
# MAIN PIPELINE
# ===========================
if __name__ == "__main__":

    # -------- Load data --------
    X_train, y_train, X_test, y_test, scaler, raw_df = prepare_data("data/sample_data.csv")

    # Split train into train/val for ARO
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    # Make global copies for optimizer
    GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN = X_tr, y_tr
    GLOBAL_X_VAL, GLOBAL_Y_VAL = X_val, y_val

    # -------- Define search space --------
    search_space = {
        "hidden_dim": (16, 64),
        "num_layers": (1, 3),
        "learning_rate": (0.0005, 0.01)
    }

    # -------- Run ARO --------
    optimizer = AROOptimizer(
        num_rabbits=6,
        search_space=search_space,
        fitness_func=fitness_function,
        iterations=5   # small number for demo
    )

    print(" Running ARO hyperparameter search...")
    best_params, best_loss = optimizer.optimize()
    print("\n Best hyperparameters found:")
    print(best_params)
    print(f"Validation loss: {best_loss:.6f}")

    import json
    os.makedirs("results", exist_ok=True)
    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best hyperparameters saved → results/best_params.json")


    # -------- Final training using best hyperparameters --------
    print("\n Training final model...")
    model = train_final_model(best_params, X_train, y_train)

    print("\n Training completed!")
