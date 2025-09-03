from typing import *

import torch
import torch.nn as nn
from functools import lru_cache


class BiGRU(nn.Module):
        def __init__(
                self, 
                hidden_size,
                num_hidden_layers, 
                vocab_size, 
                hidden_dropout_prob,
                num_classes,
                pad_idx=None,
                **kwargs
            ):
            super(BiGRU, self).__init__()
            self.hidden_size = hidden_size
            self.dropout = hidden_dropout_prob
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)

            # The GRU takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_size.
            self.gru = nn.GRU(
                input_size=hidden_size, 
                hidden_size=hidden_size, 
                num_layers=num_hidden_layers,
                batch_first=True,
                bidirectional=True, bias=True, dropout=hidden_dropout_prob
            )

            # The linear layer that maps from hidden state space to tag space
            self.fc1 = nn.Linear(
                in_features=self.hidden_size*2, out_features=self.hidden_size // 2
            )
            self.fc2 = nn.Linear(self.hidden_size // 2, num_classes)

        def init_hidden_gru(self, size):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_size)
            return torch.autograd.Variable(
                torch.zeros(self.gru.num_layers*2, size, self.hidden_size)
            )
        
        @lru_cache
        def num_parameters(self):
            n_params = sum(p.numel() for p in self.parameters())
            return n_params

        def forward_batch(self, batch:Dict):
            inputs = batch["inputs"]
            logits = self.forward(inputs)

            if 'targets' in batch:
                logits = logits.view(-1, logits.size(-1))

            return {"outputs": logits}

        def forward(self, inputs:Dict[str, Any]):
            x = inputs['input_ids']

            hidden_gru = self.init_hidden_gru(x.size(0))
            hidden_gru = hidden_gru.to(x.device)
            embeds = self.word_embeddings(x)

            gru_input = torch.nn.functional.alpha_dropout(embeds, p=self.dropout, training=self.training)
            gru_out, self.hidden_gru = self.gru(gru_input, hidden_gru)

            feat = gru_out[:, -1, :]
            feat = torch.nn.functional.dropout(feat, p=self.dropout, training=self.training)
            out = torch.relu_(self.fc1(feat))
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out = self.fc2(out)
            return out