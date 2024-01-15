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

# calculate similarity
def calculate_similarity(embs, hash_codes):
    dis_func = nn.PairwiseDistance(p=2)
    embs.cpu()
    embs_temp = embs.clone().detach().cpu()
    distances = []
    similarities = []
    # # method 1
    # similarity_cal = torch.cosine_similarity(embs_temp.unsqueeze(1), embs_temp.unsqueeze(0), dim=-1)
    # method 2
    embs_temp = embs_temp / torch.norm(embs_temp, dim=-1, keepdim=True) # 方差归一化，即除以各自的模
    similarity_cal = torch.mm(embs_temp, embs_temp.T)
    for i in range(embs.size(0)):
        similarities.append(similarity_cal[i, :].squeeze()[i:])
    return distances, similarities


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

    # comm_time_start = time.time()
    x = tree.map_structure(scatter_func, inp)
    # comm_time += time.time() - comm_time_start

    comm_time_start = time.time()
    x = expert_fn(x, fwd_expert_count)
    comm_time += time.time() - comm_time_start

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
    # comm_time += time.time() - comm_time_start
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
        self.workloads_throttling = [[] for i in range(8)]

        # calculate traffic size per batch
        self.traffic = []
        self.traffic_new = []

        # save token to expert distribution 
        self.tokens_to_experts = []

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

    def expert_fn_single(self, inp, fwd_expert_count, idx=0):
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

    def forward(self, moe_inp, original_shape, total_experts, top_k, layer_idx, fuse_token=False, training_step=0):
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
        throttling_costs = 0
        start_step =0
        num_experts = total_experts

        # print('total tokens is {}, dimension is {}'.format(moe_inp.size(0), moe_inp.size(1)))

        # # ------------------------------------------------ save gate score ------------------------------------------------ # #
        save_gate_score = False
        if save_gate_score == True:
            gate_score_save = gate_top_k_idx.clone().detach().cpu().numpy()
            if training_step == 10:
                np.savez(f'./workloads/gate_xl/gates_{layer_idx}_device{self.moe_rank}_top2.npz', gate_score_save)
        # # ----------------------------------------------------------------------------------------------------------------- # #


        # # -------------------------------------- save token to calculate similarity --------------------------------------- # #
        save_tokens = False
        if save_tokens == True:
            if training_step == 0 and layer_idx == 0:
                save_token_embeddings = moe_inp.clone().detach().cpu().numpy()
                np.savez('./workloads/transformerxl_tokens_before_experts.npz', save_token_embeddings)
        # # ----------------------------------------------------------------------------------------------------------------- # #
        # current_workloads = []
        # # # ----------------------------- token throttling with similarity (multiple threshold) ----------------------------- # #
        # if layer_idx == 0:
        #     for step, threshold in enumerate([1-(i*0.1) for i in range(10)]):
        #         token_throttling = True
        #         if token_throttling == True:
        #             moe_inp_temp = moe_inp.clone().detach()
        #             if layer_idx == 0:
        #                 gate_top_k_idx_temp = gate_top_k_idx.clone().detach()
        #                 _, similarities = calculate_similarity(moe_inp_temp)
        #                 keep_token_mask = torch.ones(moe_inp_temp.size(0), dtype=torch.bool)
        #                 for i in range(len(similarities)):
        #                     if keep_token_mask[i] == True:
        #                         similar_tokens_idx = torch.nonzero(similarities[i] >= threshold).view(-1)
        #                         similar_tokens_idx_new = similar_tokens_idx[1:].add(i)
        #                         # same gate
        #                         similar_gate_out_idx = torch.nonzero(gate_top_k_idx_temp[similar_tokens_idx_new] == gate_top_k_idx_temp[i])
        #                         if similar_gate_out_idx != torch.Size([]) and similar_tokens_idx_new.size(0) > 1:
        #                             ignore_tokens_idx = similar_tokens_idx_new[similar_gate_out_idx]
        #                             similar_tokens_idx_new = ignore_tokens_idx[1:].add(i)
        #                             keep_token_mask[ignore_tokens_idx] = 0
        #                 gate_top_k_idx_new = gate_top_k_idx_temp[keep_token_mask, :]
        # # # ----------------------------------------------------------------------------------------------------------------- # #

        # # -------------------------------------------- calculate traffic size --------------------------------------------- # #
        traffic_size = 0
        calculate_traffic_size = False
        if calculate_traffic_size == True:
            if layer_idx == 0 and self.moe_rank==0:
                print('sequence length: ', moe_inp.size(0))
            for k in range(top_k_value):
                send = torch.nonzero(gate_top_k_idx[:, k] != self.moe_rank).squeeze()
                if send.dim() != 0:
                    num_send = send.size(0)
                    traffic_size += num_send
            send_size = (traffic_size*moe_inp.size(1)*4*32*2)/(1024*1024*1024)
            self.traffic_new.append(send_size)
        # if training_step == 3 and self.moe_rank==0:
        #     print(np.mean(self.traffic_new),',')
        # # ----------------------------------------------------------------------------------------------------------------- # #


        # # --------------------------------------- token throttling with similarity ----------------------------------------- # #
        token_throttling = False
        if token_throttling == True:
            time_start = time.time()
            moe_inp_temp = moe_inp.clone().detach()
            threshold = 11
            gate_top_k_idx_temp = gate_top_k_idx.clone().detach()
            # gate as the hash codes
            hash_code = gate_top_k_idx_temp[:, 0].view(-1)
            # all compare
            # hash_code = torch.zeros(gate_top_k_idx_temp.size(0), dtype=torch.int32)
            _, similarities = calculate_similarity(moe_inp_temp, hash_code)
            keep_token_mask = torch.ones(moe_inp_temp.size(0), dtype=torch.bool)
            token_idx = np.array(range(moe_inp_temp.size(0)))
            replace_mask = torch.tensor(token_idx, dtype=torch.int64)
            send_id = 0
            for i in range(len(similarities)):
                if keep_token_mask[i] == True: # threshold 越小，for循环次数越少，因此开销越低
                    similar_tokens_idx = torch.nonzero(similarities[i] > threshold).view(-1)
                    similar_tokens_idx_new = similar_tokens_idx[1:].add(i)
                    replace_mask[i] = send_id
                    # same gate
                    similar_gate_out_idx = torch.nonzero(gate_top_k_idx_temp[similar_tokens_idx_new] == gate_top_k_idx_temp[i])
                    if similar_gate_out_idx != torch.Size([]) and similar_tokens_idx_new.size(0) > 1:
                        ignore_tokens_idx = similar_tokens_idx_new[similar_gate_out_idx]
                        similar_tokens_idx_new = ignore_tokens_idx[1:].add(i)
                        keep_token_mask[ignore_tokens_idx] = 0
                        replace_mask[ignore_tokens_idx] = send_id
                    send_id += 1
            throttling_costs = time.time() - time_start
            # gate_top_k_idx_new = gate_top_k_idx_temp[keep_token_mask, :]
                # print("similarity calculation time costs: ",time.time()-time_start)
        # # ----------------------------------------------------------------------------------------------------------------- # #

        if token_throttling == True:
            # print(replace_mask)
            moe_inp = moe_inp[keep_token_mask]
            gate_top_k_idx = gate_top_k_idx[keep_token_mask]

        # # ----------------------------------------- workloads without throttling ------------------------------------------ # #
        calculate_workloads = False
        if calculate_workloads == True and layer_idx == 0:
            for i in range(num_experts):
                workload_in_experts = 0
                for j in range(top_k_value):
                    workload_tensor = torch.nonzero(gate_top_k_idx[:, j] == i).squeeze()
                    if workload_tensor.dim() != 0:
                        num_tokens = workload_tensor.size(0)
                        workload_in_experts += num_tokens
                self.workloads[i].append(workload_in_experts)
            if training_step == 200:
                np.savez(f'./workloads/workloads_on_experts_gpt/worker_expert{self.moe_rank}.npz', self.workloads)
        # # ----------------------------------------------------------------------------------------------------------------- # #


        # # ------------------------------------------ workloads with throttling -------------------------------------------- # #
        # if token_throttling == True and layer_idx == 0 and calculate_workloads == True:
        #     for i in range(num_experts):
        #         workload_in_experts = 0
        #         for j in range(top_k_value):
        #             workload_tensor = torch.nonzero(gate_top_k_idx_new[:, j] == i).squeeze()
        #             if workload_tensor.dim() != 0:
        #                 num_tokens = workload_tensor.size(0)
        #                 workload_in_experts += num_tokens
        #         self.workloads_throttling[i].append(workload_in_experts)
        #     if training_step == 200:
        #         np.savez(f'./workloads/workloads_on_experts_gpt_throttling/worker_expert{self.moe_rank}.npz', self.workloads_throttling)
        # # ----------------------------------------------------------------------------------------------------------------- # #


        # # --------------------------------------- save token to expert distribution --------------------------------------- # #
        save_token2expert = False
        if save_token2expert == True:
            workload_in_experts = [0 for i in range(num_experts)]
            for i in range(num_experts):
                for j in range(top_k_value):
                    workload_tensor = torch.nonzero(gate_top_k_idx[:, j] == i).squeeze()
                    if workload_tensor.dim() != 0:
                        num_tokens = workload_tensor.size(0)
                        workload_in_experts[i] += num_tokens
            self.tokens_to_experts.append(workload_in_experts)
            if training_step == 25:
                np.savez(f'./workloads/workloads_on_experts_distribution_gpt/worker_layer{layer_idx}_{self.moe_rank}.npz', self.tokens_to_experts)
        # # ----------------------------------------------------------------------------------------------------------------- # #

        time_start = time.time()
        if os.environ.get('FMOE_FASTER_SCHEDULE_ENABLE') == '1':
            fwd = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )
        else:
            fwd, _ = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )
        comm_time = time.time() - time_start

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


        if token_throttling == True:
            # output_temp = torch.zeros(moe_inp.size(0), top_k_value, moe_inp.size(1), dtype=torch.float32).to(gate_top_k_idx.device)
            output_temp = moe_outp[replace_mask]
            moe_outp = output_temp

        gate_score = gate_score.view(-1, 1, self.top_k)


        # # # ------------------------------- save similarity before and after expert execution ------------------------------- # #
        # if training_step == 1:
        #     moe_inp_temp = moe_inp.clone().detach()
        #     moe_inp_temp = moe_inp_temp / torch.norm(moe_inp_temp, dim=-1, keepdim=True) # 方差归一化，即除以各自的模
        #     similarity_cal = torch.mm(moe_inp_temp, moe_inp_temp.T)
        #     save_similarity = similarity_cal.cpu().numpy()
        #     np.savez('./workloads/similarity_before_after_experts/gpt_sim_before_experts_device{}_layer{}.npz'.format(self.moe_rank, layer_idx), save_similarity)

        # # save token similarity after expert execution
        # if training_step == 1:
        #     moe_outp_temp = moe_outp[:,0,:].squeeze().clone().detach()
        #     moe_outp_temp = moe_outp_temp / torch.norm(moe_outp_temp, dim=-1, keepdim=True) # 方差归一化，即除以各自的模
        #     similarity_cal = torch.mm(moe_outp_temp, moe_outp_temp.T)
        #     save_similarity = similarity_cal.cpu().numpy()
        #     np.savez('./workloads/similarity_before_after_experts/gpt_sim_after_experts_device{}_layer{}.npz'.format(self.moe_rank, layer_idx), save_similarity)
        # # # ----------------------------------------------------------------------------------------------------------------- # #


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
        return moe_outp, throttling_costs, comm_time, 0
