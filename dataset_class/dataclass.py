import torch
import random
import pandas as pd
import configuration as configuration

from typing import Dict
from torch import Tensor
from torch.utils.data import Dataset
from dataset_class.preprocessing import tokenizing, no_multi_spaces
from dataset_class.preprocessing import adjust_sequences, subsequent_tokenizing


class QuestionDocumentMatchingDataset(Dataset):
    """ pytorch dataset module for QuestionDocumentMatching Task in metric learning
    such as contrastive loss, arcface, multiple-negative ranking loss, InfoNCE

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
        self.tokenizer_func = tokenizing
        self.questions = df["question"].tolist()
        self.contexts = df["inputs"].tolist()

    def __len__(self) -> int:
        return len(self.questions)

    def cal_seq_len(self, context: str) -> int:
        """function for calculating the input sequence's length by query encoder's pretrained tokenizer

        Args:
            context (str): text to calculate the sequence length
        """
        return self.cfg.tokenizer.encode(context, add_special_tokens=False).__len__()

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token
        question, context = no_multi_spaces(self.questions[item]), no_multi_spaces(self.contexts[item])
        prompt = f"{cls} query: " + question + f" {sep} passage: " + context + f" {sep}"

        # calculate the left context size in input prompt
        # random sampling for making negative samples of metric learning
        """
        left = self.cfg.max_len - self.cal_seq_len(prompt)
        indices = list(range(len(self.questions)))
        random.shuffle(indices)

        i = 0
        while left >= 128:
            cnt = indices[i]
            if cnt != item:
                curr = self.contexts[cnt]
                left -= self.cal_seq_len(curr)
                prompt += f" {sep} passage: {curr} {sep}"

            i += 1
        """

        # encode the input prompt
        # for input_ids, attention_mask
        batches = self.tokenizer_func(
                text=prompt,
                cfg=self.cfg,
                max_length=self.cfg.max_len,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
        for k, v in batches.items():
            batches[k] = torch.as_tensor(v)

        # find sep token index for making query index mask and document index mask
        # query tensor range: 1 ~ q_i, document tensor range: q_i+1 ~ d_i
        counter = 0
        q_i, d_i = None, []
        for i, v in enumerate(batches['input_ids']):
            if not counter and v == self.cfg.tokenizer.sep_token_id:
                q_i = i
                counter += 1

            elif counter and v == self.cfg.tokenizer.sep_token_id:
                d_i = i
                break

        batches['query_index'] = torch.as_tensor(q_i)
        batches['document_index'] = torch.as_tensor(d_i)
        return batches
