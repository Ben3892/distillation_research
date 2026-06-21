import torch
from torch import nn

from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformers.spec_utils import ModuleSpec
from megatron.core.transformers.transformers_config import TransformersConfig
from megatron.training import get_args

from megatron_patch.utils.fused_distill_cross_entropy_v2_add_jsd import(
    fused_distill_vocab_parallel_cross_entropy_v2,
    calc_student_topk_logits
)

class M3_Model(GPTModel):

    def __init__(
            self,
            config: LightningConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            **kwargs,
    ):
        self.args = get_args(config)
        self.transformer_layer_spec = transformer_layer_spec
        self.calculate_jsd_loss = self.args.jsd_loss_weight > 0
        self.calculate_mtp_jsd_loss = self.args.mtp_jsd_loss_weight > 0
        self.jsd_beta = self.args.jsd_beta
        self.temperature = self.args.temperature

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

    def _get_mtp_loss_function(self):

        if not self.args.use_distillation:

            def standard_loss_wrapper(labels, logits, **kwargs):
                ce_loss = tensor_parallel.vocab_parallel_cross_entropy_loss(
                    logits.contiguous(),
                    labels.transpose(0, 1).contiguous(),
                )
                return ce_loss
            return standard_loss_wrapper
        else:
            def fused_distill_loss_wrapper(labels, logits, teacher_topk_logits, teacher_topk_indices, **kwargs):
                ce_loss, distill_loss, jsd_loss = fused_distill_cross_entropy_v2(
                    logits.contiguous(),
                    labels.transpose(0, 1).contiguous(),
                    teacher_topk_logits,
                    teacher_topk_indices,
                    **kwargs
                )
                return ce_loss, distill_loss, jsd_loss
            return fused_distill_loss_wrapper

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input: torch.Tensor = None,
            labels: torch.Tensor = None,
            teacher_topk_logits: torch.Tensor = None,
            teacher_topk_indices: torch.Tensor = None,
            loss_mask: torch.Tensor = None,
            packed_seq_params = None,
            quality_id: torch.Tensor = None,
            **kwargs
    ):
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
            inference_context=kwargs.get("inference_context", None),
        )

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=kwargs.get("inference_context", None),
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            **kwargs.get('extra_block_kwargs', {}),
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weights()

        if self.mtp_process:
            mtp_loss_fn = self._get_mtp_loss_function()

            mtp_teacher_logits = teacher_topk_logits.transpose(0, 1).contiguous() if teacher_topk_logits is not None else None
            mtp_teacher_indices = teacher_topk_indices.transpose(0, 1).contiguous() if teacher_topk_indices is not None else None

            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weights()

            hidden_states_after_mtp = self.mtp(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                teacher_topk_logits=teacher_topk_logits,
                teacher_topk_indices=teacher_topk_indices,
                compute_language_model_loss=mtp_loss_fn,
                embedding=self.embedding,
                otuput_layer=self.output_layer,
                output_weight=output_weight,
                quality_id=quality_id,
            )
        else:
            hidden_states_after_mtp = hidden_states

        lm_logits, _ = self.output_layer(hidden_states=hidden_states_after_mtp, weights=output_weight)

        if labels is None:
            return lm_logits.transpose(0, 1).contiguous()

        lm_loss, distill_loss, jsd_loss = None, None, None

        if self.use_distillation:
            lm_loss, distill_loss, jsd_loss = fused_distill_cross_entropy_v2(
                lm_logits.contiguous(),
                labels.transpose(0, 1).contiguous(),
                teacher_topk_logits.transpose(0, 1).contiguous(),
                teacher_topk_indices.transpose(0, 1).contiguous(),
                self.calculate_jsd_loss,
                self.jsd_beta,
                self.temperature,
            )

        else:
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(
                lm_logits.contiguous(),
                labels.transpose(0, 1).contiguous(),
            )
        return (lm_loss, distill_loss, jsd_loss)

