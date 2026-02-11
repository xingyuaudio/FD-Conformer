import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_ln = self.ln(x)
        y, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        return self.drop(y)


class ConformerConvModule(nn.Module):
    """
    Conformer conv module along frequency tokens.
    x: (B,F,C)
    """
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.glu = GLU()

        self.dw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=True,
        )
        self.act = Swish()
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_ln = self.ln(x)
        y = self.pw1(x_ln)
        y = self.glu(y)           # (B,F,C)
        y = y.transpose(1, 2)     # (B,C,F)
        y = self.dw(y)            # (B,C,F)
        y = self.act(y)
        y = y.transpose(1, 2)     # (B,F,C)
        y = self.pw2(y)
        y = self.drop(y)
        return y


class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, conv_kernel, dropout, ff_scale=0.5):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ln_out = nn.LayerNorm(d_model)
        self.ff_scale = ff_scale

    def forward(self, x):
        x = x + self.ff_scale * self.ff1(x)
        x = x + self.mha(x)
        x = x + self.conv(x)
        x = x + self.ff_scale * self.ff2(x)
        x = self.ln_out(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, conv_kernel, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, d_ff, n_heads, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class ConformerBlock_NoConv(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, ff_scale=0.5):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ln_out = nn.LayerNorm(d_model)
        self.ff_scale = ff_scale

    def forward(self, x):
        x = x + self.ff_scale * self.ff1(x)
        x = x + self.mha(x)
        x = x + self.ff_scale * self.ff2(x)
        x = self.ln_out(x)
        return x


class ConformerEncoder_NoConv(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock_NoConv(d_model, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class PerFreqMLPBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class PerFreqMLPEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            PerFreqMLPBlock(d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class ConvResBlock1D(nn.Module):
    def __init__(self, d_model, k, dilation, dropout):
        super().__init__()
        assert k % 2 == 1
        pad = dilation * (k // 2)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            bias=True,
        )
        self.act = Swish()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = x.transpose(1, 2)  # (B,C,F)
        h = self.conv(h)
        h = h.transpose(1, 2)  # (B,F,C)
        h = self.drop(self.act(h))
        return x + h


class ConvEncoder1D(nn.Module):
    def __init__(self, d_model, n_layers, k, dilation, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvResBlock1D(d_model, k=k, dilation=dilation, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class LearnableFreqPosEnc(nn.Module):
    def __init__(self, n_freq, d_model, dropout=0.0):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_freq, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pos)


class SpatialMappingModule(nn.Module):
    """
    x_sparse: (B,D,2,F) treat as (B,C=D,H=2,W=F)
    """
    def __init__(self, input_dirs, output_dirs, act: nn.Module):
        super().__init__()
        # Spatial Mapping module in the paper:
        # direct sparse-to-dense mapping across directions for each frequency bin.
        self.map2 = nn.Sequential(
            nn.Conv2d(input_dirs, output_dirs, kernel_size=1, bias=True),
            act,
        )

    def forward(self, x):
        # Produces \hat{H}_{log}^{spatial} in the paper:
        # (B, D_sparse, 2, F) -> (B, D_dense, 2, F)
        return self.map2(x)


class DirectionExpansionHead(nn.Module):
    """
    (B,F,C) -> (B,793,2,F)
    """
    def __init__(self, d_model, d_hidden, n_dense, dropout):
        super().__init__()
        self.n_dense = n_dense
        self.n_ears = 2
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_dense * 2),
        )

    def forward(self, x):
        x = self.ln(x)
        y = self.net(x)  # (B,F,793*2)
        B, Freq, _ = y.shape
        y = y.view(B, Freq, self.n_dense, self.n_ears)   # (B,F,793,2)
        y = y.permute(0, 2, 3, 1).contiguous()           # (B,793,2,F)
        return y
