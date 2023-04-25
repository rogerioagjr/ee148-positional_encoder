import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class TransformerModel(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, nhead: int, d_hid: int,
                 num_layers: int, text_encoder: nn.Module, pos_encoder: nn.Module,
                 dropout: float = 0.5, init_range: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.text_encoder = text_encoder(n_tokens, d_model)
        self.pos_encoder = pos_encoder(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_tokens)

        self.init_weights(init_range=init_range)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.text_encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

