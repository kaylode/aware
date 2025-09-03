from typing import *
from pathlib import Path
import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset
from ehrpfn.utils.vocabulary import Vocabulary
from scripts.tokenization.constants import SpecialToken as ST
from traddl.source.datasets._sharded_data import ShardedData
from datetime import timedelta
from tqdm import tqdm
from theseus import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


class BaseShardedDataset(Dataset):
    def __init__(self, input_dir: str, **kwargs):
        self._data = ShardedData(Path(input_dir))

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
    def adm_time_offset(self):
        return self._data.adm_time_offset
    
    @property
    def times(self):
        return self._data.times
    
    @property
    def token_type_ids(self):
        return self._data.token_type_ids
    
    @property
    def patient_offset_at_idx(self):
        """Aka patient data start at idx."""
        return self._data.patient_offset_at_idx
    
    def _get_indices_of_stokens(self, stokens: str | Sequence[str]) -> th.Tensor:
        if isinstance(stokens, str):
            stokens = [stokens]
        tokens_of_interest = th.tensor(self.vocab.encode(stokens))
        shard_indices = []
        token_offset = 0
        for token_chunk in self.tokens:
            new_indices = th.nonzero(th.isin(token_chunk, tokens_of_interest)).view(-1)
            new_indices += token_offset
            shard_indices.append(new_indices)
            token_offset += len(token_chunk)

        return th.cat(shard_indices)

    def _match(
        self,
        ordered_sequence: th.Tensor,
        input: th.Tensor,
        *,
        fill_unmatched: int | None = None,
        shift: int = 0,
    ) -> th.Tensor:
        """Match each value in the input tensor to the corresponding value in an ordered sequence.
        
        This function finds the position of each value from the input tensor in the ordered_sequence
        tensor and returns the value at that position (with optional adjustment using shift).
        
        Used to match related token pairs (e.g., START_ICD with corresponding END_ICD tokens).
        
        Args:
            ordered_sequence: A sorted tensor of indices (e.g., positions of END_ICD tokens).
            input: A tensor of indices to match against the ordered sequence (e.g., positions of START_ICD tokens).
            fill_unmatched: Value to use for elements in input that don't have a match in ordered_sequence.
                            If None, will use the closest value in ordered_sequence.
            shift: Integer offset to apply to the matched indices (useful for off-by-one adjustments).
            
        Returns:
            A tensor where each element from input is matched to its corresponding element
            in ordered_sequence based on position, with optional shift and handling of unmatched values.
            
        Example:
            # Match each START_ICD token index to its corresponding END_ICD token index
            start_indices = self._get_indices_of_stokens([ST.START_ICD])
            end_indices = self._get_indices_of_stokens([ST.END_ICD])
            matched_end_indices = self._match(end_indices, start_indices)
        """
        ordered_sequence_indices = th.searchsorted(ordered_sequence, input, right=True)
        if shift:
            ordered_sequence_indices += shift
        if fill_unmatched is None:
            return ordered_sequence[ordered_sequence_indices]
        else:
            out = th.full_like(input, fill_value=fill_unmatched)
            mask = ordered_sequence_indices < len(ordered_sequence)
            out[mask] = ordered_sequence[ordered_sequence_indices[mask]]
            return out

    

class PatientOutcomeDataset(BaseShardedDataset):
    def __init__(
            self, 
            input_dir:str, 
            n_positions:int=64,
            vocab_path:str=None,
            padding_side:str='left',
            append_cls: bool=False,
            split: str = "train",
            **kwargs
        ):
        super().__init__(input_dir, **kwargs)
        self.append_cls = append_cls
        self.num_classes = 2
        self.timeline_size = n_positions
        self.padding_side = padding_side
        self.split = split
        
        self.vocab = Vocabulary.from_path(vocab_path)
        self.vocab_size = len(self.vocab.stoi.keys())
        self.start_indices = self._get_indices_of_stokens(ST.ADMISSION)

    def __len__(self) -> int:
        return len(self.start_indices)

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
    
    
    def __getitem__(self, idx: int) -> dict[str, th.Tensor | dict[str, th.LongTensor]]:

        start_id = int(self.start_indices[idx].item())
        patient_id = self.patient_id_at_idx[start_id]
        hadm_id = self._data.hadm_id[start_id]
        label = self._data.labels[start_id]

        end_idx = self.patient_data_end_at_idx[start_id]
        
        if isinstance(end_idx, th.Tensor):
            end_idx = end_idx.item()

        timeline_size = start_id + self.timeline_size
        if self.append_cls:
            timeline_size -= 1 #CLS token

        # Query data from sharded tensors
        code = self.tokens[start_id : min(timeline_size , end_idx)]
        ages = self.patient_age[start_id : min(timeline_size, end_idx)]
        ages = th.nan_to_num(ages)
        ages[ages < 0] = 0

        times = self.adm_time_offset[start_id : min(timeline_size, end_idx)]
        token_type_ids = self.token_type_ids[start_id : min(timeline_size, end_idx)]
        segments = self.generate_seq_based_on_sep(code, symbol=self.vocab.stoi[ST.ADMISSION])
        visit_order = self.generate_visit_orderings(code, symbol=self.vocab.stoi[ST.ADMISSION])

        # Append CLS token to the beginning of the code
        if self.append_cls:
            code = th.cat([th.LongTensor([self.vocab.stoi[ST.CLS]]),code ])

        # Pad code sequence
        pad_index = self.vocab.stoi[ST.PAD]
        tokens = th.cat([code, th.zeros(self.timeline_size - len(code), dtype=th.long) + pad_index])

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
            'hadm_id': hadm_id,
            'inputs': {
                'input_ids': th.LongTensor(tokens),
                'attention_mask': th.LongTensor(mask),
                'position_ids': th.LongTensor(position),
                'segment_ids': th.LongTensor(segments),
                'time_stamps': th.LongTensor(times),
                'age': th.LongTensor(ages),
                'type_ids': th.LongTensor(token_type_ids),
                'visit_order': th.LongTensor(visit_order),
            },
            'targets': label.long()
        }
 
    def collate_fn(self, batch):
        batch_size = len(batch)
        if isinstance(batch, dict):
            return batch

        inputs = {
            'patient_ids': th.stack([b['patient_id'] for b in batch]),
            'hadm_ids': th.stack([b['hadm_id'] for b in batch]),
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
    
    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes. For balance sampling
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []

        for i in tqdm(range(self.__len__())):
            label = self._data.labels[i]
            self.classes_dist.append(int(label.item()))

        return self.classes_dist
