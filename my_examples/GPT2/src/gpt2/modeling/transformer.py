import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from gpt2.utils.fusing import LayerNorm
from gpt2.modeling import (PadMasking, FutureMasking, AttentionLayer, Past,
                           PositionalEmbedding, TokenEmbedding,
                           PositionwiseFeedForward)
from typing import Optional, Tuple, List, Union


from fmoe.transformer import FMoETransformerMLP
class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, moe_num_expert=64, moe_world_size=1, moe_group=None, moe_top_k=2):
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, world_size=moe_world_size, moe_group=moe_group,
            top_k=moe_top_k, activation=activation)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, fuse_token=False, train_step=0):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out, fusion_costs, comm_time = super().forward(self.layer_norm(inp), fuse_token, train_step)
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out, fusion_costs, comm_time = super().forward(inp, fuse_token, train_step)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output, fusion_costs, comm_time

class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 dropout: float = 0.1,
                 moe: bool = False, 
                 moe_num_expert: int = 64, 
                 moe_world_size: int = 1, 
                 moe_group = None,
                 moe_top_k: int = 2):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)

        self.moe = moe

        if moe is False:
            self.ff = PositionwiseFeedForward(dims, rate, dropout)
        else:
            self.ff = CustomizedMoEPositionwiseFF(dims, dims * rate, dropout,
                                                 moe_world_size=moe_world_size,
                                                 moe_group=moe_group,
                                                 pre_lnorm=False,
                                                 moe_num_expert=moe_num_expert,
                                                 moe_top_k=moe_top_k)

        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)


    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                fuse_token: bool = False,
                train_step: int = 0,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)

        x = x + a
        if self.moe is False:
            x = x + self.ff(self.ln_ff(x))
            return x if self.training else (x, past)
        else:
            o, fusion_costs, comm_time = self.ff(self.ln_ff(x), fuse_token, train_step)
            x = x + o
            return x if self.training else (x, past), fusion_costs, comm_time


class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 layers: int,
                 pad_idx: int,
                 words: int,
                 seq_len: int,
                 heads: int,
                 dims: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 moe: bool = False, 
                 moe_num_expert: int = 64, 
                 moe_world_size: int = 1, 
                 moe_group = None,
                 moe_top_k: int = 2, 
                 fuse_token: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, dims, rate, dropout, moe, moe_num_expert, moe_world_size, moe_group, moe_top_k)
            for _ in range(layers)])
        self.ln_head = LayerNorm(dims)

        self.moe = moe
        self.fuse_token =fuse_token
        if self.moe is True:
            self.total_fusion_costs = 0
            self.total_comm_time = 0
        

    def forward(self,
                x: torch.Tensor,
                past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False,
                train_step: int = 0,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0

        if self.moe is True:
            self.total_fusion_costs = 0
            self.total_comm_time = 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint,
                                      transformer)
            if self.moe is False:
                x = transformer(x, past[i] if past is not None else None, mask)
            else:
                x, fusion_costs, comm_time = transformer(x, past[i] if past is not None else None, mask, fuse_token = self.fuse_token, train_step = train_step)
                self.total_fusion_costs += fusion_costs
                self.total_comm_time += comm_time

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)

        if self.moe is False:
            return x if self.training else (x, present)
        else:
            return x if self.training else (x, present), self.total_fusion_costs
