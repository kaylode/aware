from typing import *
from pathlib import Path
import pickle
import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset
from traddl.source.datasets._sharded_data import ShardedData
from ehrpfn.utils.vocabulary import Vocabulary
from preprocess.longitudinal.tokenization.constants import SpecialToken as ST

from traddl.source.datasets.masking import ConceptMasker

class MaskedLanguageModelDataset(Dataset):
    def __init__(
            self, 
            input_dir:str, 
            n_positions:int=64,
            select_ratio:float=1.0,
            masking_ratio:float=0.8,
            replace_ratio:float=0.1,
            ignore_special_tokens:bool=True,
            masking_target_idx:int=-100,
            vocab_path:str=None,
            append_cls: bool = False,
            **kwargs
        ):
        self.masking_target_idx = masking_target_idx
        self._data = ShardedData(Path(input_dir))
        self.timeline_size = n_positions #CLS token
        self.append_cls = append_cls
        
        self.vocab = Vocabulary.from_path(vocab_path)
        self.vocab_size = len(self.vocab.stoi)

        if ignore_special_tokens:
            special_tokens_ids = [
                v for k,v in self.vocab.stoi.items() 
                if k.startswith("[") or k.startswith('TIME//')
            ]
        else:
            special_tokens_ids = []

        self.masker = ConceptMasker(
            self.vocab.stoi, select_ratio, 
            masking_ratio, replace_ratio, 
            ignore_tokens_ids=special_tokens_ids, 
            masking_target_idx=masking_target_idx
        )

    @property
    def patient_ids(self) -> th.Tensor:
        return th.cat([shard["patient_ids"] for shard in self._data.shards])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(len={len(self):,}, "
            f"patient_num={len(self.patient_ids):,}, "
            f"vocab_size={len(self.vocab):,})"
        )

    def __len__(self) -> int:
        return len(self.tokens) - self.timeline_size
    
    @property
    def patient_id_at_idx(self):
        return self._data.patient_id_at_idx
    
    @property
    def tokens(self):
        return self._data.tokens
    
    @property
    def patient_data_end_at_idx(self):
        return self._data.patient_data_end_at_idx
    
    @property
    def patient_age(self):
        return self._data.age_offset
    
    @property
    def times(self):
        return self._data.times
    
    @property
    def token_type_ids(self):
        return self._data.token_type_ids
    
    @property
    def adm_time_offset(self):
        return self._data.adm_time_offset
    
    @property
    def patient_offset_at_idx(self):
        """Aka patient data start at idx."""
        return self._data.patient_offset_at_idx
    
    def generate_seq_based_on_sep(self, tokens, symbol=ST.SEP):
        flag = 1
        seg = []
        for token in tokens:
            if token == symbol:
                seg.append(flag)
                if flag == 1:
                    flag = 2
                else:
                    flag = 1
            else:
                seg.append(flag)
        return th.LongTensor(seg)


    def generate_visit_orderings(self, tokens, symbol=ST.SEP):
        pos = []
        flag = 1
        for token in tokens:
            if token == symbol:
                pos.append(flag)
                flag += 1
            else:
                pos.append(flag)
        return th.LongTensor(pos)
    
    def __getitem__(self, idx: int) -> tuple[th.Tensor | tuple, th.Tensor]:
        patient_id = self.patient_id_at_idx[idx]
        admission_id = self._data.admission_id[idx]
        hadm_id = self._data.hadm_id[idx]

        end_idx = self.patient_data_end_at_idx[idx]

        timeline_size = idx + self.timeline_size 
        if self.append_cls:
            timeline_size -= 1 #CLS token

        # Query data from sharded tensors
        code = self.tokens[idx : min(timeline_size , end_idx)]
        ages = self.patient_age[idx : min(timeline_size, end_idx)]
        ages = th.nan_to_num(ages)
        ages[ages < 0] = 0

        times = self.adm_time_offset[idx : min(timeline_size, end_idx)]
        token_type_ids = self.token_type_ids[idx : min(timeline_size, end_idx)]
        segments = self.generate_seq_based_on_sep(code, symbol=self.vocab.stoi[ST.ADMISSION])
        visit_order = self.generate_visit_orderings(code, symbol=self.vocab.stoi[ST.ADMISSION])

        code, label = self.masker.mask_patient_concepts(code)

        # Append CLS token to the beginning of the code
        if self.append_cls:
            code = th.cat([th.LongTensor([self.vocab.stoi[ST.CLS]]),code ])
            label = th.cat([th.zeros(1, dtype=th.long)+self.masking_target_idx, label]) # label for the first token

        # Pad code sequence
        pad_index = self.vocab.stoi[ST.PAD]
        tokens = th.cat([code, th.zeros(self.timeline_size - len(code), dtype=th.long) + pad_index])
        label = th.cat([label, th.zeros(self.timeline_size - len(label), dtype=th.long) + self.masking_target_idx])

        # Masking
        mask = np.ones(self.timeline_size)
        # based on pad index
        mask[tokens == pad_index] = 0
            
        # Append Zero to the beginning of the age
        if self.append_cls:
            ages = th.cat([th.zeros(1, dtype=th.long), ages]) # time diff from some reference (e.g. admission time)
            times = th.cat([th.zeros(1, dtype=th.long), times]) # time diff from some reference (e.g. admission time)
            token_type_ids = th.cat([th.zeros(1, dtype=th.long), token_type_ids]) 
            segments = th.cat([th.zeros(1, dtype=th.long), segments]) # even and odd numbering to distinguish between two segments
            visit_order = th.cat([th.zeros(1, dtype=th.long), visit_order]) # subsequent visits increasing order

        # Pad age sequence and code sequence
        ages = th.cat([ages, th.zeros(self.timeline_size - len(ages), dtype=th.long)]).long()
        times = th.cat([times, th.zeros(self.timeline_size - len(times), dtype=th.long)])
        token_type_ids = th.cat([token_type_ids, th.zeros(self.timeline_size - len(token_type_ids), dtype=th.long)])
        visit_order = th.cat([visit_order, th.zeros(self.timeline_size - len(visit_order), dtype=th.long)])
        segments = th.cat([segments, th.zeros(self.timeline_size - len(segments), dtype=th.long)])
        position = th.arange(0, self.timeline_size, dtype=th.long) # traditional position ids
        
        return {
            'patient_id': patient_id,
            'admission_id': admission_id,
            'inputs': {
                'input_ids': th.LongTensor(tokens),
                'attention_mask': th.LongTensor (mask),
                'position_ids': th.LongTensor(position),
                'segment_ids': th.LongTensor(segments),
                'time_stamps': th.LongTensor(times),
                'age': th.LongTensor(ages),
                'type_ids': th.LongTensor(token_type_ids),
                'visit_order': th.LongTensor(visit_order),
            },
            'targets': label
        }

    def collate_fn(self, batch):
        batch_size = len(batch)
        if isinstance(batch, dict):
            return batch
        
        inputs = {
            'input_ids': th.stack([b['inputs']['input_ids'] for b in batch]),
            "type_ids": th.stack([b['inputs']['type_ids'] for b in batch]),
            'time_stamps': th.stack([b['inputs']['time_stamps'] for b in batch]),
            'ages': th.stack([b['inputs']['age'] for b in batch]),
            'visit_orders': th.stack([b['inputs']['visit_order'] for b in batch]),
            'visit_segments': th.stack([b['inputs']['segment_ids'] for b in batch]),
            'attention_mask': th.stack([b['inputs']['attention_mask'] for b in batch]),
            'position_ids': th.stack([b['inputs']['position_ids'] for b in batch]),
        }
        targets = th.stack([b['targets'] for b in batch])
        return {
            'inputs': inputs,
            'targets': targets
        }       