"""MTTS-CAN (Multi-Task Temporal Shift Convolutional Attention Network)
NeurIPS 2020
- Two-stream CAN with Temporal Shift Module
- Multi-task heads: BVP (heart) + Respiration

This file mirrors TS_CAN.py style but exposes only MTTS_CAN class
to be imported by the model factory.
"""

# neural_methods/model/MTTS_CAN.py
import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    # normalize by sum over HxW and scale by 0.5*H*W
    def forward(self, x):
        # x: [N*T, 1, H, W], assumed already passed through sigmoid
        s = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True).clamp_min(1e-6)
        h, w = x.size(2), x.size(3)
        return x / s * (0.5 * h * w)


class _TSM(nn.Module):
    # temporal shift (fold_div=3 to match TS-CAN style)
    def __init__(self, n_segment: int, fold_div: int = 3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):  # [N*T, C, H, W]
        nt, c, h, w = x.shape
        t = self.n_segment
        n = nt // t
        x = x.view(n, t, c, h, w)
        fold = c // self.fold_div
        out = x.clone()
        if fold > 0:
            out[:, 1:, :fold] = x[:, :-1, :fold]               # forward
            out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]   # backward
            # remaining channels unchanged
        return out.view(nt, c, h, w)


class MTTS_CAN(nn.Module):
    """
    input : [N*T, 6, H, W] = [diff_rgb(3) | raw_rgb(3)]
            if 3ch only -> treat as raw; diff is approximated inside
    output: (bvp, resp) each [N*T, 1]
    """
    def __init__(
        self,
        in_channels: int = 3,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        kernel_size: int = 3,
        dropout_mid: float = 0.25,
        dropout_head: float = 0.5,
        pool_size: int = 2,
        nb_dense: int = 128,
        frame_depth: int = 10,
        img_size: int = 72,
    ):
        super().__init__()
        self.T = frame_depth
        act = nn.Tanh()

        # motion stream (TSM only here)
        self.tsm1 = _TSM(frame_depth)
        self.tsm2 = _TSM(frame_depth)
        self.tsm3 = _TSM(frame_depth)
        self.tsm4 = _TSM(frame_depth)

        self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
        self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)
        self.motion_act = act

        # appearance stream
        self.appearance_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
        self.appearance_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
        self.appearance_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
        self.appearance_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)
        self.appearance_act = act

        # attention heads from appearance
        self.appearance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1, bias=True)
        self.appearance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1, bias=True)
        self.mask1 = Attention_mask()
        self.mask2 = Attention_mask()

        # pooling / dropout
        self.pool = nn.AvgPool2d(pool_size)
        self.drop_mid = nn.Dropout(dropout_mid)
        self.drop_y = nn.Dropout(dropout_head)
        self.drop_r = nn.Dropout(dropout_head)

        # heads (flatten size auto)
        self.fc1_y = nn.LazyLinear(nb_dense)
        self.fc2_y = nn.Linear(nb_dense, 1)
        self.fc1_r = nn.LazyLinear(nb_dense)
        self.fc2_r = nn.Linear(nb_dense, 1)

    # split [N*T,C,H,W] → (diff, raw)
    def _split_motion_raw(self, x):
        if x.size(1) == 6:
            return x[:, :3], x[:, 3:]
        if x.size(1) == 3:
            nt, c, h, w = x.shape
            n = nt // self.T
            x5 = x.view(n, self.T, c, h, w)
            diff = (x5[:, 1:] - x5[:, :-1]) / (x5[:, 1:] + x5[:, :-1] + 1e-6)
            diff = torch.cat([diff, diff[:, -1:]], dim=1).reshape(nt, c, h, w)
            return diff, x
        raise ValueError(f"unsupported channel size: {x.size(1)}")

    # window-mean raw → broadcast along time
    def _appearance_avg_broadcast(self, raw):
        nt, c, h, w = raw.shape
        n = nt // self.T
        raw5 = raw.view(n, self.T, c, h, w)
        avg1 = raw5.mean(dim=1)                               # [N,3,H,W]
        rep = avg1.unsqueeze(1).repeat(1, self.T, 1, 1, 1)    # [N,T,3,H,W]
        return rep.view(nt, c, h, w)

    def forward(self, x):                                     # [N*T, C, H, W]
        motion, raw = self._split_motion_raw(x)
        raw = self._appearance_avg_broadcast(raw)

        # shallow
        d = self.tsm1(motion)
        d = self.motion_act(self.motion_conv1(d))
        r = self.appearance_act(self.appearance_conv1(raw))

        d = self.tsm2(d)
        d = self.motion_act(self.motion_conv2(d))
        r = self.appearance_act(self.appearance_conv2(r))

        # attention 1 (conv → sigmoid → mask)
        g1 = torch.sigmoid(self.appearance_att_conv1(r))
        g1 = self.mask1(g1)
        d = d * g1
        d = self.pool(d); d = self.drop_mid(d)
        r = self.pool(r); r = self.drop_mid(r)

        # deep
        d = self.tsm3(d); d = self.motion_act(self.motion_conv3(d))
        d = self.tsm4(d); d = self.motion_act(self.motion_conv4(d))
        r = self.appearance_act(self.appearance_conv3(r))
        r = self.appearance_act(self.appearance_conv4(r))

        # attention 2
        g2 = torch.sigmoid(self.appearance_att_conv2(r))
        g2 = self.mask2(g2)
        d = d * g2

        # head
        d = self.pool(d)
        d = torch.flatten(d, 1)

        y = torch.tanh(self.fc1_y(d)); y = self.drop_y(y); y = self.fc2_y(y)
        r = torch.tanh(self.fc1_r(d)); r = self.drop_r(r); r = self.fc2_r(r)
        return y, r