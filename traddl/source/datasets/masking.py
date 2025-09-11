import torch
from typing import Dict, Tuple, List
import numpy as np
from theseus import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


class ConceptMasker:
    def __init__(self, vocabulary: dict, 
                 select_ratio: float, masking_ratio: float = 0.8, replace_ratio: float=0.1,
                 ignore_tokens_ids: List[int]=[],
                 masking_target_idx:int=-100) -> None:
        """Mask concepts for MLM.
        Args:
            vocabulary: Vocabulary
            select_ratio: Ratio of tokens to consider in the loss
            masking_ratio: Ratio of tokens to mask
            replace_ratio: Ratio of tokens to replace with random word
        """
        
        self.masking_target_idx = masking_target_idx
        self.vocabulary = vocabulary
        # self.n_special_tokens = len([token for token in vocabulary if token.startswith("[")]) if ignore_special_tokens else 0
        self.select_ratio = select_ratio # Select ratio of tokens to consider in the loss
        self.masking_ratio = masking_ratio
        self.replace_ratio = replace_ratio
        self.ignore_tokens_ids = ignore_tokens_ids

        all_ids = torch.arange(len(self.vocabulary))
        # Create a mask for ignored IDs
        ignore_mask = torch.zeros_like(all_ids, dtype=torch.bool)
        # Filter ignore_tokens_ids to ensure they are within the vocabulary bounds
        valid_ignore_ids = [idx for idx in self.ignore_tokens_ids if 0 <= idx < len(self.vocabulary)]
        if valid_ignore_ids:
            ignore_mask[valid_ignore_ids] = True
        # Store the tensor of valid (non-ignored) token IDs
        self.valid_ids = all_ids[~ignore_mask]
        
        assert self.select_ratio <= 1.0, "Select ratio should not exceed 1.0"
        assert self.masking_ratio + self.replace_ratio <= 1.0, "Masking ratio + replace ratio should not exceed 1.0"
        LOGGER.text(f"Select ratio: {self.select_ratio}, masking ratio: {self.masking_ratio}, replace ratio: {self.replace_ratio},\
                    unchanged ratio: {1.0 - self.masking_ratio - self.replace_ratio}", level=LoggerObserver.INFO)

    def mask_patient_concepts(
        self, concepts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = concepts.clone()
        probability_vector = torch.full(target.shape, self.select_ratio)

        # Ignore special tokens
        # special_token_mask = concepts < self.n_special_tokens
        # probability_vector.masked_fill_(special_token_mask, value=0.0)


        # Use ignore_tokens_ids instead of checking against n_special_tokens
        ignore_mask = torch.zeros_like(concepts, dtype=torch.bool)
        for token_id in self.ignore_tokens_ids:
            ignore_mask |= (concepts == token_id)
        probability_vector.masked_fill_(ignore_mask, value=0.0)


        # Get MLM mask
        selected_indices = torch.bernoulli(probability_vector).bool()
        target[~selected_indices] = self.masking_target_idx  # Ignore loss for non-selected tokens

        # Replace with [MASK]
        indices_mask = (
            torch.bernoulli(torch.full(target.shape, self.masking_ratio)).bool()
            & selected_indices
        )
        concepts[indices_mask] = self.vocabulary["[MASK]"]

        # Replace with random word
        replace_ratio = self.replace_ratio / (
            1 - self.masking_ratio
        )  # Account for already masked tokens
        indices_replace = (
            torch.bernoulli(torch.full(target.shape, replace_ratio)).bool()
            & selected_indices
            & ~indices_mask
        )

        # Use valid_ids for random replacement instead of using n_special_tokens
        if len(self.valid_ids) > 0:
            random_indices = torch.randint(0, len(self.valid_ids), target.shape, dtype=torch.long)
            random_words = self.valid_ids[random_indices]
            concepts[indices_replace] = random_words[indices_replace]

        # random_words = torch.randint(
        #     self.n_special_tokens, len(self.vocabulary), target.shape, dtype=torch.long
        # )
        # concepts[indices_replace] = random_words[indices_replace]

        return concepts, target
    
