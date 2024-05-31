import torch
import pandas as pd
import configuration

from torch import Tensor
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """ Custom Dataset for Pretraining Task in NLP, such as MLM, CLM, ... etc

    if you select clm, dataset will be very long sequence of text, so this module will deal the text from the sliding window of the text
    you can use this module for generative model's pretrain task

    Also you must pass the input, which is already tokenized by tokenizer with cutting by model's max_length
    We recommend to use the full max_length inputs for the better performance like roberta, gpt2, gpt3 ...

    Args:
        inputs: inputs from tokenizing by tokenizer, which is a dictionary of input_ids, attention_mask, token_type_ids
        is_valid: if you want to use this dataset for validation, you can set this as True, default is False
    """
    def __init__(self, inputs: Dict, is_valid: bool = False) -> None:
        super().__init__()
        self.inputs = inputs
        self.input_ids = inputs['input_ids']
        self.is_valid = is_valid

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        batch_inputs = {}
        for k, v in self.inputs.items():
            # reduce memory usage by defending copying tensor
            batch_inputs[k] = torch.as_tensor(v[item]) if not self.is_valid else torch.as_tensor(v[item][0:2048])
        return batch_inputs


class TextGenerationDataset(Dataset):
    """ Pytorch Dataset Module for Text Generation Task in fine-tuning
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class QuestionAnsweringDataset(Dataset):
    """ Pytorch Dataset Module for QuestionAnswering Task in fine-tuning
    """

    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class QuestionDocumentMatchingDataset(Dataset):
    """ Pytorch Dataset Module for QuestionDocumentMatching Task in fine-tuning

    Args:


    """
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass
