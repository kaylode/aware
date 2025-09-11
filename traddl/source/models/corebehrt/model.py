"""
CoreBEHRT source: https://github.com/FGA-DIKU/EHR
"""

import torch
import torch.nn as nn
from transformers import ModernBertModel, ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from .embeddings import (
    TIME2VEC_ABSPOS_SCALE,
    TIME2VEC_ABSPOS_SHIFT,
    TIME2VEC_AGE_SCALE,
    TIME2VEC_AGE_SHIFT,
    EhrEmbeddings
)
from .heads import FineTuneHead


class CorebehrtEncoder(ModernBertModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=getattr(config, "age_scale", TIME2VEC_AGE_SCALE),
            age_shift=getattr(config, "age_shift", TIME2VEC_AGE_SHIFT),
            abspos_scale=getattr(config, "abspos_scale", TIME2VEC_ABSPOS_SCALE),
            abspos_shift=getattr(config, "abspos_shift", TIME2VEC_ABSPOS_SHIFT),
        )

    def forward(self, batch: dict):
        attention_mask = torch.ones_like(batch["input_ids"])
        inputs_embeds = self.embeddings(
            input_ids=batch["input_ids"],
            segments=batch["visit_segments"],
            age=batch["ages"],
            abspos=batch["position_ids"],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )


class CorebehrtForPretraining(CorebehrtEncoder):
    def __init__(
            self, 
            vocab_size: int,
            padding_idx: int,
            cls_idx: int,
            sep_idx: int,
            hidden_size: int= 96,
            num_hidden_layers: int = 3,
            num_attention_heads: int = 3,
            intermediate_size: int = 64,
            type_vocab_size: int = 1024, # !!! type_vocab_size should be > truncation_len//2 if sep token else >truncation len
            embedding_dropout: float = 0.1, 
            max_position_embeddings: int = 512, # the longest sequence length we will use this model with
            age_scale: float = 1e-2, # 100 -> 1
            abspos_scale: float = 1e-5, # 400k -> 4
            age_shift: int = -50, # center around 0
            abspos_shift: int = -90_000, # 0 will be around 2015
            sparse_pred_ignore_index: int = -100,
            **kwargs
        ):


        config = ModernBertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            type_vocab_size=type_vocab_size,
            embedding_dropout=embedding_dropout,
            max_position_embeddings=max_position_embeddings,
            age_scale=age_scale,
            abspos_scale=abspos_scale,
            age_shift=age_shift,
            abspos_shift=abspos_shift,
            vocab_size=vocab_size,
            pad_token_id=padding_idx,
            cls_token_id=cls_idx,
            sep_token_id=sep_idx,
            sparse_pred_ignore_index=sparse_pred_ignore_index,
            sparse_prediction=True,
        )
        

        super().__init__(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

    def forward_batch(self, batch: dict):
        return self.forward(batch)

    # Inspiration from ModernBertForMaskedLM
    def forward(self, batch: dict):
        outputs = super().forward(batch['inputs'])
        last_hidden_state = outputs[0]

        # labels = batch.get("targets")
        # if self.sparse_prediction and labels is not None:
        #     # flatten labels and output first
        #     labels = labels.view(-1)
        #     last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

        #     # then filter out the non-masked tokens
        #     mask_tokens = labels != self.sparse_pred_ignore_index
        #     last_hidden_state = last_hidden_state[mask_tokens]
        #     labels = labels[mask_tokens]

        logits = self.decoder(self.head(last_hidden_state))
        
        if batch.get("targets") is not None:
            logits = logits.view(-1, logits.size(-1))

        return {
            'outputs': logits,
        }

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class CorebehrtForFineTuning(CorebehrtEncoder):
    def __init__(
            self,
            vocab_size: int,
            padding_idx: int,
            cls_idx: int,
            sep_idx: int,
            num_classes: int = 2,
            hidden_size: int= 96,
            num_hidden_layers: int = 3,
            num_attention_heads: int = 3,
            intermediate_size: int = 64,
            type_vocab_size: int = 1024, # !!! type_vocab_size should be > truncation_len//2 if sep token else >truncation len
            embedding_dropout: float = 0.1, 
            max_position_embeddings: int = 512, # the longest sequence length we will use this model with
            age_scale: float = 1e-2, # 100 -> 1
            abspos_scale: float = 1e-5, # 400k -> 4
            age_shift: int = -50, # center around 0
            abspos_shift: int = -90_000, # 0 will be around 2015
            sparse_pred_ignore_index: int = -100,
            freeze_backbone: bool = False,
            **kwargs
    ):
        
        config = ModernBertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            type_vocab_size=type_vocab_size,
            embedding_dropout=embedding_dropout,
            max_position_embeddings=max_position_embeddings,
            age_scale=age_scale,
            abspos_scale=abspos_scale,
            age_shift=age_shift,
            abspos_shift=abspos_shift,
            vocab_size=vocab_size,
            pad_token_id=padding_idx,
            cls_token_id=cls_idx,
            sep_token_id=sep_idx,
            sparse_pred_ignore_index=sparse_pred_ignore_index,
            sparse_prediction=True,
        )

        super().__init__(config)

        # Freeze the encoder
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        self.cls = FineTuneHead(
            hidden_size=config.hidden_size, 
            num_classes=num_classes
        )
        for param in self.cls.parameters():
            param.requires_grad = True
        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index


    def forward_batch(self, batch: dict):
        return self.forward(batch)
    
    def forward(self, batch: dict):
        outputs = super().forward(batch['inputs'])

        sequence_output = outputs[0]  # Last hidden state
        attention_mask = batch['inputs'].get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(batch['inputs']["input_ids"])
        logits = self.cls(sequence_output, attention_mask)

        if batch.get("targets") is not None:
            logits = logits.view(-1, logits.size(-1))

        return {
            'outputs': logits,
        }
