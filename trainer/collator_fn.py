import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence


class CollatorFunc(nn.Module):
    def __init__(self, plm_cfg) -> None:
        super(CollatorFunc, self).__init__()
        self.plm_cfg = plm_cfg

    def get_mask(self, seq_len: int, query_indices: Tensor, context_indices: Tensor = None) -> Tensor:
        """ function for making the query mask and document mask for splitting the two different types of sub-sequence,
        (query and document)
        """
        mask = torch.zeros(seq_len, self.plm_cfg.hidden_size)
        if context_indices is None:  # for query mask
            mask[1:query_indices, ...] = 1
        elif context_indices is not None:  # for document mask
            mask[query_indices + 1:context_indices, ...] = 1
        return mask

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        labels = torch.as_tensor([x["labels"] for x in batched])
        input_ids = pad_sequence([x["input_ids"] for x in batched], batch_first=True, padding_value=0)
        attention_mask = pad_sequence([x["attention_mask"] for x in batched], batch_first=True, padding_value=0)
        query_mask = pad_sequence(
            [self.get_mask(len(x["input_ids"]), x["query_index"]) for x in batched],
            batch_first=True,
            padding_value=0
        )
        document_mask = pad_sequence(
            [self.get_mask(len(x["input_ids"]), x["query_index"], x["document_index"]) for x in batched],
            batch_first=True,
            padding_value=0
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "query_mask": query_mask,
            "document_mask": document_mask
        }
