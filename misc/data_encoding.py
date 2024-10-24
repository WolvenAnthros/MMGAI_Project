from base_imports import *

"""
Positional encoding with MultiHeadAttention layers takes too much time to calculate
SO that simple positional encoding that is presented in data_collection.py is used
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        # Multi-Head Self-Attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1,
                 fixed_output_size=16):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.fixed_output_size = fixed_output_size
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((fixed_output_size, fixed_output_size))

    def forward(self, x):
        x = self.input_proj(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (h * w, b, c)

        x = self.pos_encoder(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.permute(1, 2, 0).view(b, c, h, w)
        x = self.adaptive_pool(x)
        return x


if __name__ == "__main__":
    fixed_output_size = 16

    encoder = TransformerEncoder(in_channels=1, d_model=64, nhead=8, num_layers=6, fixed_output_size=fixed_output_size)

    input_tensor1 = torch.randn(1, 1, 32, 32)  # 32x32 input
    input_tensor2 = torch.randn(1, 1, 64, 64)  # 64x64 input
    input_tensor3 = torch.randn(1, 1, 128, 128)  # 128x128 input

    output_tensor1 = encoder(input_tensor1)
    output_tensor2 = encoder(input_tensor2)
    output_tensor3 = encoder(input_tensor3)

    print(f"Output tensor 1 shape: {output_tensor1.shape}")
    print(f"Output tensor 2 shape: {output_tensor2.shape}")
    print(f"Output tensor 3 shape: {output_tensor3.shape}")
