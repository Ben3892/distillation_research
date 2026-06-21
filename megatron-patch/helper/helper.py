import os
import torch
import time
from functools import partial
from collections import deque

from megatron.core import mpu
from megatron.training import get_args, get_timers
from megatron.core.num_batches_calculator import get_num_microbatches
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_tp_rank,
    get_batch_on_this_cp_rank
)
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSequenceParams
from megatron.core.transformers.multi_token_prediction import MTPLossLoggingHelper

from megatron_patch.data.utils import (
    get_batch_on_this_tp_rank_original,
    get_batch_on_this_tp_rank_idxmap_sft,
    get_batch_on_this_tp_rank_with_teacher_quality_id,
)
from megatron.core import parallel_state
from megatron_patch.teacher import TeacherClient

from sglang.test.registered.quant.test_fp8_kernel import device
from transformers.examples.pytorch.context_parallel import position_ids

teacher_client = None
train_batch_buffer = deque()
valid_batch_buffer = deque()
log_cnt = 0

def get_batch(data_iterator, is_eval=False):

    args = get_args()
    if args.use_distillation:
        if args.teacher_type == "local":
            return get_batch_base(data_iterator)
        return get_batch_with_teacher_knowledge(data_iterator, is_eval=is_eval)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, _, _ = get_batch_base(data_iterator)
        return tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, None, None

def get_batch_base(data_iterator):

    args = get_args()
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None, None, None

    if args.dataset == "PRETRAIN-WITH-WEIGHT":
        if args.train_mode != "pretrain":
            raise ValueError('the PRETRAIN-WITH-WEIGHT dataset should only be used for pretrain mode')
        batch = get_batch_on_this_tp_rank(data_iterator)
        num_seqs = batch.pop("num_seqs", None)
        batch = get_batch_on_this_cp_rank(data_iterator)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None,
            batch['dataset_name'],
            batch['tokens'].cpu() if 'tokens_cpu' not in batch else batch['tokens'],
        )
    elif args.datset == "PT-TEACHER":
        if args.train_mode != "pretrain":
            raise ValueError('the PT-TEACHER dataset should only be used for pretrain mode')
        batch = get_batch_with_teacher_knowledge(data_iterator)
        num_seqs = batch.pop("num_seqs", None)
        batch = get_batch_on_this_cp_rank(data_iterator)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            batch['dataset_name'],
            batch.get('quality_id', None),
            batch['teacher_topk_logps'],
            batch['teacher_topk_indices'],
        )
    else:
        raise ValueError('the dataset should be PT-TEACHER or PRETRAIN-WITH-WEIGHT')

def get_batch_with_teacher_knowledge(data_iterator, is_eval=False):
    args = get_args()
    global teacher_client

    def is_teacher_client_rank():
        return mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0
    num_microbatches = get_num_microbatches()
    if is_teacher_client_rank() and teacher_client is None:
        if args.teacher_type =="real":
            teacher_ips = [ip.strip() for ip in args.teacher_ips.split(',') if ip.strip()]
            dp_rank = parallel_state.get_data_parallel_rank()
            teacher_ip = teacher_ips[dp_rank%len(teacher_ips)]
            print(f"dp_rank: {dp_rank}, teacher_ip: {teacher_ip}")
            assert num_microbatches > 0
            teacher_client = TeacherClient(
                server_ip=teacher_ip,
                server_port=args.teacher_server_port,
                num_microbatches=num_microbatches,
                num_server_workers=2,
                temperature=1.0,
            )
        else:
            assert False, "distillation must note teacher ip"

        batch_buffer = train_batch_buffer
        if is_eval:
            batch_buffer = valid_batch_buffer
        n_prefills_max = 3 * get_num_microbatches()
        n_prefills = 1 if batch_buffer else n_prefills_max
        begin_index =  args.consumed_train_samples

        if n_prefills == 1:
            begin_index += n_prefills_max * args.micro_batch_size
        for np_index in range(n_prefills):
            batch_data = get_batch_base(data_iterator)
            if is_teracher_client_rank():
                per_dp_batch_size = args.global_batch_size // mpu.get_data_parallel_world_size()
                if n_prefills == n_prefills_max:
                    local_sample_index = begin_index + (np_index + 1) * args.micro_batch_size
                else:
                    local_sample_index = begin_index + args.global_batch_size
                rank = mpu.get_data_parallel_rank()
                data_index = f"rank{rank}_bs{per_dp_batch_size}_sample{local_sample_index}"

                tokens = batch_data[-1]
                dataset_name = batch_data[-2]
                teacher_knowledge_future = teacher_client.submit(tokens, dataset_name, data_index)
            else:
                teacher_knowledge_future = None
            batch_buffer.append(batch_data + (teacher_knowledge_future,))

        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, dataset_name, _, teacher_knowledge_future = batch_buffer.popleft()

        teacher_topk_logits, teacher_topk_indices = None, None
        if mpu.is_pipeline_last_stage():
            if is_teacher_client_rank():
                _, teacher_topk_logits, teacher_topk_indices = teacher_knowledge_future.result()

                teacher_topk_logits = teacher_topk_logits.cuda(non_blocking=True).to(torch.bfloat16)
                teacher_topk_indices = teacher_topk_indices.cuda(non_blocking=True).to(torch.int32)

            else:
                topk = 256
                shape = (args.micro_batch_size, args.seq_length, topk)
                teacher_topk_logits = torch.empty(*shape, dtype=torch.bfloat16, device=torch.cuda.current_device())
                teacher_topk_indices = torch.empty(*shape, dtype=torch.int32, device=torch.cuda.current_device())

            torch.distributed.broadcast(teacher_topk_logits, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
            torch.distributed.broadcast(teacher_topk_indices, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
        return tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, teacher_topk_logits, teacher_topk_indices

def loss_func(
        loss_mask: torch.Tensor,
        num_seqs: torch.Tensor,
        quality_id: torch.Tensor,
        output_from_model: tuple
):
    args = get_args()

    lm_loss, distill_loss, jsd_loss = output_from_model

    loss_mask_view = loss_mask.transpose(0, 1).contiguous().view(-1).float()
    loss_mask_view = loss_mask_view.view(-1)

    if loss_mask_view.sum() == 0:
        zero_loss = torch.tensor(0.0, device=loss_mask_view.device)

        if num_seqs is None:
            return zero_loss, {"lm loss": torch.tensor(0.0)}
        else:
            return zero_loss, torch.tensor(0, device=num_seqs.device), {"lm loss": torch.tensor(0.0)}
    losses_dict = {}
    if quality_id is not None and lm_loss.dim()==2:
        quality_weight = quality_id.unsqueeze(0).float()
        lm_quality_weight = torch.where(quality_weight==1.0, 1.0, args.lm_weight)
        raw_lm_loss_sum = torch.sum(lm_loss.view(-1) * loss_mask_view)
        lm_loss = lm_loss * lm_quality_weight

    else:
        quality_weight = None
        raw_lm_loss_sum = None

    reduced_lm_loss = torch.stack([torch.sum(lm_loss.view(-1) * loss_mask_view), loss_mask_sum])

    reduced_distill_loss = None
    raw_distill_loss_sum = None

    if args.use_distillation and distill_loss is not None:
        distill_loss_sum, raw_distill_loss_sum = selective_distill_loss(
            torch.nan_to_num(distill_loss),
            loss_mask_view,
            quality_weight
        )
        reduced_distill_loss = torch.stack([distill_loss_sum, raw_distill_loss_sum])

    reduced_jsd_loss = None
    raw_jsd_loss_sum = None
    if args.use_distillation and jsd_loss is not None:
        jsd_loss_sum, raw_jsd_loss_sum = selective_jsd_loss(
            torch.nan_to_num(jsd_loss),
            loss_mask_view,
            quality_weight
        )
        reduced_jsd_loss = torch.stack([jsd_loss_sum, raw_jsd_loss_sum])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(reduced_lm_loss, group=mpu.get_context_parallel_group())
        if reduced_distill_loss is not None:
            torch.distributed.all_reduce(reduced_distill_loss, group=mpu.get_context_parallel_group())
        if reduced_jsd_loss is not None:
            torch.distributed.all_reduce(reduced_jsd_loss, group=mpu.get_context_parallel_group())
    losses_dict["lm_loss"] = (raw_lm_loss_sum if raw_lm_loss_sum is not None else reduced_lm_loss[0] / (reduced_lm_loss[1] + 1e-8)).detach()

    if args.use_distillation and distill_loss is not None:
        log_distill = raw_distill_loss_sum if raw_distill_loss_sum is not None else reduced_distill_loss[0]
        losses_dict["distill loss"] = (log_distill / (reduced_distill_loss[1] + 1e-8)).detach()
    if args.use_distillation and jsd_loss is not None:
        log_jsd = raw_jsd_loss if raw_jsd_loss is not None else reduced_jsd_loss[0]
        losses_dict["jsd loss"] = (log_jsd / (raw_jsd_loss[1] + 1e-8)).detach()

    if args.mtp_num_layers is not None:
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=1.0,
            itereation=None,
            writer=None,
            wandb_writer=None,
            total_loss_dict=losses_dict
        )

    total_loss_sum = torch.tensor(0.0, device=loss_mask.device)
    if reduced_lm_loss is not None:
        total_loss_sum += reduced_lm_loss[0] * args.lm_loss_weight
    if args.use_distillation and distill_loss is not None:
        total_loss_sum += distill_loss[0] * args.distillation_loss_weight * (args.temperature ** 2)
    if args.use_distillation and jsd_loss is not None:
        total_loss_sum += jsd_loss[0] * args.jsd_loss_weight * (args.temperature ** 2)

    if num_seqs is None:
        total_loss_avg_token = total_loss_sum / reduced_lm_loss[1]
        return total_loss_avg_token * args.context_parallel_size, losses_dict
    else:
        return total_loss_sum * args.context_parallel_size, num_seqs.sum(), losses_dict

def forward_step(data_iterator, model):

    timers = get_timers()
    args = get_args()
    timers("batch-generator", log_level=2).start()
    start_time = time.time()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, quality_id, \
    teacher_topk_logits, teacher_topk_indices = get_batch(data_iterator, not model.training)
    timers("batch-generator", log_level=2).stop()
    end_time = time.time()

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        packed_seq_params=packed_seq_params,
        loss_mask=loss_mask,
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
        quality_id=quality_id,
    )
    return output_tensor, partial(loss_func, loss_mask, num_seqs, quality_id)
