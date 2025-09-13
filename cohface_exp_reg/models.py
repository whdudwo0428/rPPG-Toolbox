# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SeqRegressor(nn.Module):
    """RR-only: 입력 C, 출력 1"""
    def __init__(self, input_dim=16, hidden=128, layers=2, cell='lstm', bidir=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        self.bidir = bidir
        rnn_cls = nn.LSTM if cell.lower()=='lstm' else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers>1 else 0.0, bidirectional=bidir)
        out_dim = hidden*(2 if bidir else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, lengths=None):
        # x: [B,T,C]
        out, _ = self.rnn(x)  # [B,T,H*dir]
        y = self.head(out)    # [B,T,1]
        return y
