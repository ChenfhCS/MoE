r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import FMoE
from .linear import FMoELinear
from .fastermoe.config import switch_from_env

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        world_size=1,
        moe_group=None,
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        def one_expert(d_model):
            return _Expert(1, d_model, d_hidden, activation, rank=0)
        
        expert = one_expert
        # print("moe world size: ", world_size)
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, world_size=world_size, moe_group=moe_group, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

        self.total_experts = num_expert * world_size
        self.top_k = kwargs.get('top_k')

    def forward(self, inp: torch.Tensor, layer_idx = 0, fuse_token=False, train_step=0):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        # print("change successful: ", fuse_token)
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output, fusion_costs, comm_time, traffic_size = super().forward(inp, original_shape, self.total_experts, self.top_k, layer_idx = layer_idx, fuse_token=fuse_token, train_step=train_step)
        # return output.reshape(original_shape), fusion_costs, comm_time, traffic_size
        return output.reshape(original_shape), fusion_costs
