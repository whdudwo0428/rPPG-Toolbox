# -*- coding: utf-8 -*-
import torch.nn as nn


class SeqRegressor(nn.Module):
    def __init__(self, in_dim=16, hidden=128, layers=2, bidir=True, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):  # x: [B,T,16]
        y, _ = self.rnn(x)
        y = self.head(y)  # [B,T,1]
        return y