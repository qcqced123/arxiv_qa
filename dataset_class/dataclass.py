import torch
import pandas as pd
import configuration

from torch import Tensor
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from preprocessing import tokenizing, no_multi_spaces


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
    """ pytorch dataset module for QuestionDocumentMatching Task in metric learning such as SimCLR, ArcFace, MNRL

    workflow:
        1) load dataframe: question-document relation dataset
        2) drop the empty rows, which is not containing the question
        3) lower the text for the better performance
        4) remove multi spaces

    Args:
        df: pd.DataFrame, dataframe containing [paper_id, doc_id, doc, doc embedding, question]
            (doc_id is role of label, indicating the question-document pair relationship_

    Returns:
        batch_inputs: Dict, dictionary containing the query and document inputs
    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        super().__init__()
        self.cfg = cfg
        self.df = df[df['question'].notnull()]  # remove empty question rows
        self.tokenizer = tokenizing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        batch_inputs = {
            'query': self.df.at[item, 'question'],
            'document': self.df.at[item, 'doc']
        }

        for key, text in batch_inputs.items():
            text = no_multi_spaces(text.lower())
            text = self.tokenizer(
                text=text,
                cfg=self.cfg,
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )

            for k, v in text.items():
                text[k] = torch.as_tensor(v)

            batch_inputs[key] = text

        return batch_inputs
