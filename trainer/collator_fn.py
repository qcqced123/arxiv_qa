import torch.nn as nn

from torch import Tensor
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence


class CollatorFunc(nn.Module):
    def __init__(self) -> None:
        super(CollatorFunc, self).__init__()

    @staticmethod
    def forward(batched: List[Dict[str, Tensor]]) -> Dict:
        labels = [x["labels"] for x in batched]
        query_index = [x["query_index"] for x in batched]
        document_index = [x["document_index"] for x in batched]
        input_ids = pad_sequence([x["input_ids"] for x in batched], batch_first=True, padding_value=0)
        attention_mask = pad_sequence([x["attention_mask"] for x in batched], batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "query_index": query_index,
            "document_index": document_index
        }