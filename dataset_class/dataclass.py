import torch
import pandas as pd
import configuration as configuration

from torch import Tensor
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from dataset_class.preprocessing import tokenizing, no_multi_spaces
from dataset_class.preprocessing import adjust_sequences, subsequent_tokenizing


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


class QuestionDocumentMatchingDataset(Dataset):
    """ pytorch dataset module for QuestionDocumentMatching Task in metric learning
    such as contrastive loss, arcface, multiple-negative ranking loss

    workflow:
        1) load dataframe: question-document relation dataset
        2) drop the empty rows, which is not containing the question
        3) lower the text for the better performance
        4) remove multi spaces

    Args:
        df: pd.DataFrame, dataframe containing [paper_id, doc_id, doc, doc embedding, question]
            (label is role of label, indicating the question-document pair relationship)

    Returns:
        batch_inputs: Dict, dictionary containing the query and document inputs
    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizing
        self.questions = df['question'].tolist()
        self.documents = df['doc'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token

        query = subsequent_tokenizing(
            self.cfg,
            no_multi_spaces(self.questions[item])
        )
        document = subsequent_tokenizing(
            self.cfg,
            no_multi_spaces(self.documents[item])
        )

        sequences, _ = adjust_sequences(
            [query, document],
            self.cfg.max_len-5
        )

        query, document = [self.cfg.tokenizer.decode(seq) for seq in sequences]
        prompt = f"{cls} " + query + f" {sep} " + document + f" {sep}"
        batches = self.tokenizer(
                text=prompt,
                cfg=self.cfg,
                max_length=self.cfg.max_len,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )

        print(len(batches["input_ids"]))

        # for input_ids, attention_mask
        for k, v in batches.items():
            batches[k] = torch.as_tensor(v)

        # find sep token index for making query index mask and document index mask
        # query tensor range: 1 ~ q_i, document tensor range: q_i+1 ~ d_i
        counter = 0
        q_i, d_i = None, None
        for i, v in enumerate(batches['input_ids']):
            if not counter and v == self.cfg.tokenizer.sep_token_id:
                q_i = i
                counter += 1

            elif counter and v == self.cfg.tokenizer.sep_token_id:
                d_i = i
                break

        batches['query_index'] = torch.as_tensor(q_i)
        batches['document_index'] = torch.as_tensor(d_i)
        batches['labels'] = torch.as_tensor(self.labels[item])
        return batches


