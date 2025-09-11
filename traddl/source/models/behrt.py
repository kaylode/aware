from typing import *
import os
from functools import lru_cache
import numpy as np
import pytorch_pretrained_bert as Bert
import torch
import torch.nn as nn


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, **config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act','gelu'),
            hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
            max_position_embeddings = config.get('max_position_embeddings'),
            initializer_range=config.get('initializer_range', 0.02),
        )
        self.seg_vocab_size = 2
        self.age_vocab_size = 120

    def save_pretrained(self, folder):
        """ Save a configuration object to the specified directory"""
        os.makedirs(folder, exist_ok=True)
        config_file = os.path.join(folder, 'config.json')
        with open(config_file, 'w') as f:
            f.write(self.to_json_string())
        
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)
        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, batch,
                output_all_encoded_layers=True):
        
        age_ids = batch['inputs'].get('age_ids', None)
        seg_ids = batch['inputs'].get('segment_ids', None)
        posi_ids = batch['inputs'].get('position_ids', None)
        attention_mask = batch['inputs'].get('attention_mask', None)
        input_ids = batch['inputs']['input_ids']

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMaskedLM(Bert.modeling.BertPreTrainedModel):
    def __init__(self, **kwargs):
        config = BertConfig(**kwargs)
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    @lru_cache
    def num_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)
    
    def forward_batch(self, batch:Dict):
        sequence_output, _ = self.bert(batch,
                                       output_all_encoded_layers=False)
        logits = self.cls(sequence_output)

        if 'targets' in batch:
            logits = logits.view(-1, logits.size(-1))
        
        return {
            'outputs': logits
        }
    

class BEHRT(Bert.modeling.BertPreTrainedModel):
    def __init__(
            self, 
            num_classes:int,
            **kwargs
        ):
        config = BertConfig(**kwargs)
        super(BEHRT, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_bert_weights)

    @lru_cache
    def num_parameters(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)
    
    def forward_batch(self, batch:Dict):
        sequence_output, _ = self.bert(batch,
                                       output_all_encoded_layers=False)
        logits = self.cls(sequence_output)[:, 0, :]


        if 'targets' in batch:
            logits = logits.view(-1, logits.size(-1))
        
        return {
            'outputs': logits
        }
        