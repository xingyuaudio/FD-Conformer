import torch
import torch.nn as nn

from .blocks import (
    LearnableFreqPosEnc, SpatialMappingModule, DirectionExpansionHead,
    ConformerEncoder, ConformerEncoder_NoConv, PerFreqMLPEncoder,
    ConvEncoder1D,
)


class HRTFMagUpsampler(nn.Module):
    """
    Input : x_sparse (B,D,2,F)
    Output: y_pred   (B,793,2,F)
    """
    def __init__(
        self,
        n_sparse: int,
        n_dense: int,
        n_freq: int,
        variant: str,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        conv_kernel: int,
        head_hidden: int,
        dropout: float,
        use_freq_pe: bool,
        spatial_act: str,
    ):
        super().__init__()
        self.n_sparse = n_sparse
        self.n_dense = n_dense
        self.n_freq = n_freq
        self.variant = variant

        # Activation used in the Spatial Mapping module
        if spatial_act == "identity":
            act = nn.Identity()
        elif spatial_act == "silu":
            act = nn.SiLU(inplace=True)
        elif spatial_act == "relu":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown spatial_act: {spatial_act}")

        # Spatial Mapping module (paper Sec. "Spatial Mapping Module")
        self.spatial_mapping = SpatialMappingModule(n_sparse, n_dense, act=act)

        in_dim = n_sparse * 3  # [L, R, ILD]
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
        )

        # Frequency positional encoding (paper: learnable frequency positional encoding)
        self.fpos = LearnableFreqPosEnc(n_freq, d_model, dropout) if use_freq_pe else nn.Identity()

        if variant == "conformer":
            self.encoder = ConformerEncoder(d_model, d_ff, n_heads, n_layers, conv_kernel, dropout)
        elif variant == "conformer_wo_conv":
            self.encoder = ConformerEncoder_NoConv(d_model, d_ff, n_heads, n_layers, dropout)
        elif variant == "mlp":
            self.encoder = PerFreqMLPEncoder(d_model, d_ff, n_layers, dropout)
        elif variant == "conv":
            self.encoder = ConvEncoder1D(d_model, n_layers, k=3, dilation=1, dropout=dropout)
        elif variant == "dilated_b2":
            self.encoder = ConvEncoder1D(d_model, n_layers, k=3, dilation=2, dropout=dropout)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Direction expansion head in the paper (maps frequency-domain features to dense directions)
        self.head = DirectionExpansionHead(d_model, head_hidden, n_dense, dropout)

    def forward(self, x_sparse: torch.Tensor) -> torch.Tensor:
        B, D, E, Freq = x_sparse.shape
        assert D == self.n_sparse and E == 2 and Freq == self.n_freq

        # Spatial Mapping module output: \hat{H}_{log}^{spatial}
        y_base = self.spatial_mapping(x_sparse)  # (B,793,2,F)

        x = x_sparse.permute(0, 3, 1, 2).contiguous()  # (B,F,D,2)
        # Binaural Spectral Representation (paper Sec. "Binaural Spectral Representation"):
        # for each frequency bin we form [L, R, L-R]
        L = x[..., 0]      # (B,F,D)
        R = x[..., 1]      # (B,F,D)
        ILD = L - R        # (B,F,D)  ≈ (L - R) in the paper

        tok = torch.cat([L, R, ILD], dim=-1).contiguous()  # (B,F,3D)  -> \mathbf{S} in the paper

        # Frequency-Domain Modeling module with Conformer blocks
        h = self.in_proj(tok)                              # (B,F,C)   latent spectral space
        h = self.fpos(h)
        h = self.encoder(h)
        y_res = self.head(h)                               # (B,793,2,F) = \hat{H}_{log}^{freq}

        # Final output: \hat{H}_{log} = \hat{H}_{log}^{spatial} + \hat{H}_{log}^{freq}
        return y_base + y_res
