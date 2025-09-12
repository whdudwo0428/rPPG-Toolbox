
import torch.nn as nn

class SeqRegressor(nn.Module):
    def __init__(self, input_dim=8, hidden=128, layers=2, bidir=True, cell="LSTM", dropout=0.0):
        super().__init__()
        rnn_cls = nn.GRU if cell.upper() == "GRU" else nn.LSTM
        do = (float(dropout) if layers > 1 else 0.0)  # PyTorch RNN dropout은 layers>1일 때만 적용
        self.rnn = rnn_cls(input_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=do)
        h_out = hidden * (2 if bidir else 1)
        # 멀티헤드: [rr, hr]
        self.head = nn.Linear(h_out, 2)
    def forward(self, x):
        y,_ = self.rnn(x); y = self.head(y); return y
