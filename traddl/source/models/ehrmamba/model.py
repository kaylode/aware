"""
Source: https://github.com/VectorInstitute/Odyssey
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import (
    MambaCausalLMOutput,
    MambaForCausalLM,
)

from .mamba_utils import (
    MambaForSequenceClassification,
    MambaSequenceClassifierOutput,
)
from functools import lru_cache

class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False, time_format: str = "seconds"):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> Any:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(
        self,
        visit_order_size: int,
        embedding_size: int,
    ):
        super().__init__()
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> Any:
        """Apply visit embedding to the input visit segments."""
        return self.embedding(visit_segments)

class MambaEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from concept, token_type, etc., embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: MambaConfig,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initiate wrapper class for embeddings used in Mamba CEHR classes."""
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(
            self.type_vocab_size,
            config.hidden_size,
        )
        self.visit_order_embeddings = nn.Embedding(
            self.max_num_visits,
            config.hidden_size,
        )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=config.hidden_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size + 2 * time_embeddings_size,
            config.hidden_size,
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # End copy

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids_batch: Optional[torch.Tensor] = None,
        time_stamps: Optional[torch.Tensor] = None,
        ages: Optional[torch.Tensor] = None,
        visit_orders: Optional[torch.Tensor] = None,
        visit_segments: Optional[torch.Tensor] = None,
    ) -> Any:
        """Return the final embeddings of concept ids.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input data (concept_ids) to be embedded.
        inputs_embeds : torch.Tensor
            The embeddings of the input data.
        token_type_ids_batch : torch.Tensor
            The token type IDs of the input data.
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Using cached values from a prior cache_input call
        time_stamps_embeds = self.time_embeddings(time_stamps)
        ages_embeds = self.age_embeddings(ages)
        visit_segments_embeds = self.visit_segment_embeddings(visit_segments)
        visit_order_embeds = self.visit_order_embeddings(visit_orders)
        token_type_embeds = self.token_type_embeddings(token_type_ids_batch)

        inputs_embeds = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds),
            dim=-1,
        )

        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = inputs_embeds + token_type_embeds
        embeddings += visit_order_embeds
        embeddings += visit_segments_embeds

        embeddings = self.dropout(embeddings)

        return self.LayerNorm(embeddings)

class MambaPretrain(nn.Module):
    """Mamba model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        use_mambapy: bool = False,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.use_mambapy = use_mambapy

        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
            use_mambapy=self.use_mambapy,
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        )
        # Initialize weights and apply final processing
        self.post_init()

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @lru_cache
    def num_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward_batch(self, batch):
        outputs = self.forward(batch['inputs'])
        logits = outputs.logits
        if 'targets' in batch:
            logits = logits.view(-1, logits.size(-1))
        return {
            'outputs': logits,
        }

    def forward(
        self,
        batch: Dict[str, Any],
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""

        inputs = (
            batch["input_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )

        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        return self.model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class MambaFinetune(nn.Module):
    """Mamba model for fine-tuning."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        use_mambapy: bool = False,
        classifier_dropout: float = 0.1,
        problem_type: str = "single_label_classification",
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.use_mambapy = use_mambapy

        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
            use_mambapy=self.use_mambapy,
            num_labels=num_classes,
            classifier_dropout=classifier_dropout,
            problem_type = problem_type
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        )

        # Freeze the encoder
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False


        self.model = MambaForSequenceClassification(
            config=self.config, 
            freeze_backbone=freeze_backbone
        )

        self.post_init()

        


    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

    @lru_cache
    def num_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward_batch(self, batch):
        outputs = self.forward(batch['inputs'])
        logits = outputs.logits
        if 'targets' in batch:
            logits = logits.view(-1, logits.size(-1))
        return {
            'outputs': logits,
        }

    def forward(
        self,
        batch: Dict[str, Any],
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaSequenceClassifierOutput]:
        """Forward pass for the model."""

        inputs = (
            batch["input_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )

        # labels = batch["labels"]
        # task_indices = batch["task_indices"]

        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        return self.model(
            input_ids=concept_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )