r"""
FMoE core layer
"""
import time
import tree
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from .functions import prepare_forward, ensure_comm
from .functions import MOEScatter, MOEGather
from .functions import AllGather, Slice
from .gates import NaiveGate

from .fastermoe.config import switch_from_env

def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    comm_time = 0
    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )

    comm_time_start = time.time()
    x = tree.map_structure(scatter_func, inp)
    # comm_time += time.time() - comm_time_start


    x = expert_fn(x, fwd_expert_count)

    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )

    # comm_time_start = time.time()
    outp = tree.map_structure(gather_func, x)
    comm_time += time.time() - comm_time_start
    return outp, comm_time


fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `mp_group` is a deprecated alias of `slice_group`
    * `moe_group` stands for the group of process that performs expert
    parallelism. The default value `None` means all processes. See the
    parallelism document for more details of the groups.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

        if moe_group is not None:
            self.moe_rank = dist.get_rank(group=mp_group)
        else:
            self.moe_rank = 0
        
        # calculate workloads
        self.workloads = [[] for i in range(8)]
        self.measure_step = 0 # update per 12 steps, i.e., every first layer

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def expert_fn_single(self, inp, fwd_expert_count, idx):
        r"""
        forward single expert for smart scheduling.
        """
        assert not self.experts_fused, "should not use fused experts"
        output = self.experts[idx](inp, fwd_expert_count)
        return output

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp, original_shape, total_experts, top_k, layer_idx, fuse_token=False, train_step=0):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"



        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp)
        # print(self.gate,gate_top_k_idx, gate_score)
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        top_k_value = top_k
        time_costs = 0
        start_step =0
        num_experts = total_experts
        print(gate_top_k_idx.size())
        # save gate score
        gate_score_save = gate_top_k_idx.clone().detach().cpu().numpy()
        if self.measure_step == 10:
            np.savez(f'./workloads/gate_xl/gates_{layer_idx}_device{self.moe_rank}.npz', gate_score_save)
        self.measure_step += 1

        # calculate the traffic size
        traffic_size = 0
        save_traffic = []
        for k in range(top_k_value):
            send = torch.nonzero(gate_top_k_idx[:, k] != self.moe_rank).squeeze()
            if send.dim() != 0:
                num_send = send.size(0)
                traffic_size += num_send
        save_traffic.append(traffic_size*moe_inp.size(1))

        # # calculate workloads
        # for i in range(num_experts):
        #     workload_in_experts = 0
        #     for j in range(top_k_value):
        #         workload_tensor = torch.nonzero(gate_top_k_idx[:, k] == i).squeeze()
        #         if workload_tensor.dim() != 0:
        #             num_tokens = workload_tensor.size(0)
        #             workload_in_experts += num_tokens
        #     self.workloads[i].append(workload_in_experts)
        # if self.measure_step == 200:
        #     np.savez(f'./workloads/workloads_on_experts_gpt/worker_layer{layer_idx}_expert{self.moe_rank}.npz', self.workloads)
        # self.measure_step += 1
        
        # # save tokens before experts execution
        # if self.measure_step == 0 and layer_idx == 0:
        #     save_token_embeddings = moe_inp.clone().detach().cpu().numpy()
        #     np.savez('./workloads/transformerxl_tokens_before_experts.npz', save_token_embeddings)
        # self.measure_step += 1

        # token fusions
        if fuse_token == True and train_step > start_step:
            time_start = time.time()
            gate_top_k_idx_temp = gate_top_k_idx.clone().detach().to(gate_top_k_idx.device)
            # fuse inputs and gates
            batch_size = original_shape[1]
            num_token_per_input = original_shape[0]
            # output_temp = moe_inp.clone().detach().unsqueeze(1)
            output_temp = torch.zeros(moe_inp.size(0), top_k_value, moe_inp.size(1), dtype=torch.float32).to(gate_top_k_idx.device)
            # fused_input = torch.zeros(batch_size*num_experts, moe_inp.size(1), dtype=torch.float32)
            # fused_gate = torch.zeros(batch_size*num_experts, 1, dtype=torch.int64)
        
            fused_input = torch.zeros(num_experts, moe_inp.size(1), dtype=torch.float32)
            fused_input_k = torch.zeros(num_experts, top_k_value, moe_inp.size(1), dtype = torch.float32)
            fused_gate = torch.zeros(num_experts, top_k_value, dtype=torch.int64)
            # fused_gate_score = torch.zeros(batch_size*num_experts, 1, dtype=torch.float32)
            # for i in range(batch_size):
                # gate_per_input = gate_top_k_idx[i*num_token_per_input:(i+1)*num_token_per_input]
                # for j in range(num_experts):
                #     mask = torch.nonzero(gate_per_input[:,0]==j).squeeze()
                #     if mask.nelement() > 0:
                #         mask = mask+(i*num_token_per_input)
                #         fused_input[i*num_experts+j, :] = torch.mean(moe_inp[mask, :], dim=0)
                #         fused_gate[i*num_experts+j, :] = j
            # fuse all tokens in the same experts
            for j in range(num_experts):
                for k in range(top_k_value):
                    mask = torch.nonzero(gate_top_k_idx[:,k]==j).squeeze()
                    if mask.nelement() > 0:
                        fused_input_k[j, k, :] = torch.mean(moe_inp[mask, :], dim=0)
                    fused_gate[j, k] = j
                fused_input[j, :] = torch.mean(fused_input_k[j, :, :], dim=0)

            # print(fused_input)
            moe_inp = fused_input.to(moe_inp.device)
            gate_top_k_idx = fused_gate.to(gate_top_k_idx.device)
            time_costs += time.time() - time_start

        fwd, comm_time = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        # print('output size: ', moe_outp.size())
        # recovery outputs
        if fuse_token == True and train_step > start_step:
            time_start = time.time()
            # for i in range(batch_size):
            #     gate_per_input = gate_top_k_idx_temp[i*num_token_per_input:(i+1)*num_token_per_input]
            #     for j in range(num_experts):
            #         mask = torch.nonzero(gate_per_input[:, 0]==j).squeeze()
            #         if mask.nelement() > 0:
            #             mask = mask + (i*num_token_per_input)
            #             output_temp[mask, 0, :] = moe_outp[i*num_experts+j, 0, :]
            #             output_temp[mask, 1:, :] = moe_outp[i*num_experts+j, 1, :]

            for j in range(num_experts):
                for k in range(top_k_value):
                    mask = torch.nonzero(gate_top_k_idx_temp[:, k]==j).squeeze()
                    if mask.nelement() > 0:
                        output_temp[mask, k, :] = moe_outp[j, k, :]

            moe_outp = output_temp
            gate_top_k_idx = gate_top_k_idx_temp
            time_costs += time.time() - time_start
        # print("output size: ", moe_outp.size())

        gate_score = gate_score.view(-1, 1, self.top_k)

        # # save token embeddings after expert execution
        # if self.measure_step == 1 and layer_idx == 0:
        #     save_token_embeddings = moe_outp.clone().detach().cpu().numpy()
        #     np.savez('./workloads/transformerxl_tokens_after_experts.npz', save_token_embeddings)
        # self.measure_step += 1

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        # print('the communication in a forward layer is: ', comm_time)
        return moe_outp, time_costs, comm_time, traffic_size
