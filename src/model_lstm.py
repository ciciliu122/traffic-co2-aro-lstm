import torch
import torch.nn as nn

class MultiTaskLSTM(nn.Module):
    """
    Multi-output LSTM model for CO2 forecasting.
    Input shape:  (batch, seq_len, 7)
    Output shape: (batch, 7)
    """
    def __init__(self, input_dim=7, hidden_dim=32, num_layers=2, output_dim=7, dropout=0.1):
        super(MultiTaskLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected layer for multi-output regression
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 7)
        lstm_out, _ = self.lstm(x)   # lstm_out: (batch, seq_len, hidden)
        last_step = lstm_out[:, -1, :]  # take last time step
        out = self.fc(last_step)        # regression to 7 outputs
        return out
