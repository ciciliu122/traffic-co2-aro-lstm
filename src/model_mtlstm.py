import torch
import torch.nn as nn

class MultiTaskLSTM(nn.Module):
    """
    Multi-Task LSTM for CO₂ forecasting
    Shared encoder + task-specific decoders (7 sectors)
    """

    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=7, dropout=0.2):
        super(MultiTaskLSTM, self).__init__()

        # Shared LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Task-specific decoders (one per sector)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(output_dim)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        encoded_output, _ = self.encoder(x)   # (batch, seq_len, hidden)
        last_step = encoded_output[:, -1, :]  # take last step output

        outputs = []
        for decoder in self.decoders:
            y_hat = decoder(last_step)
            outputs.append(y_hat)

        # concat into shape (batch, 7)
        outputs = torch.cat(outputs, dim=1)
        return outputs
