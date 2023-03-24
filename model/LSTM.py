import torch
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.lstm_input_dim = input_dim
        self.lstm_hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0,
        )

        self.fc_input_dim = hidden_dim*2
        self.fc = nn.Linear(self.fc_input_dim, out_steps * input_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, lstm_input_dim=1)
        x = x.transpose(1, 2).contiguous()  # (batch_size, num_nodes, in_steps, 1)
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_nodes, self.in_steps, self.lstm_input_dim)

        out, _ = self.lstm(x)
        out = out[
                :, -1, :
            ]  # (batch_size * num_nodes, hidden_dim) use last step's output

        out = self.fc(out).view(
            batch_size, self.num_nodes, self.out_steps, self.lstm_input_dim
        )

        return out.transpose(1, 2)