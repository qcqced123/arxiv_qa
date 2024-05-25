import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Any
from ..tuner.mlm import WholeWordMaskingCollator
from configuration import CFG

""" Tokenizer Type for Span Boundary Objective Task
Because Span Masking Algorithm is based on WholeWordMasking, we need to know what kind of tokenizer is used.
"""
BPE = [
    'RobertaTokenizerFast',
    'GPT2TokenizerFast',
]

SPM = [
    'DebertaV2TokenizerFast',
    'DebertaTokenizerFast',
    'XLMRobertaTokenizerFast',
]

WORDPIECE = [
    'BertTokenizerFast',
    'ElectraTokenizerFast',
]


def random_non_negative_integer(max_value: int) -> int:
    return random.randint(0, max_value)


class SpanCollator(WholeWordMaskingCollator):
    """ Custom Collator for Span Boundary Objective Task, which is used for span masking algorithm
    Span Masking is similar to Whole Word Masking, but it has some differences:
        1) Span Masking does not use 10% of selected token left & 10% of selected token replaced other vocab token
            - just replace all selected token to [MASK] token
    Algorithm:
    1) Select 2 random tokens from input tokens for spanning
    2) Calculate relative position embedding for each token with 2 random tokens froms step 1.
    3) Calculate span boundary objective with 2 random tokens from step 1 & pos embedding from step 2.
    Args:
        cfg: configuration.CFG
        masking_budget: masking budget for Span Masking
                        (default: 0.15 => Recommended by original paper)
        span_probability: probability of span length for Geometric Distribution
                         (default: 0.2 => Recommended by original paper)
        max_span_length: maximum span length of each span in one batch sequence
                         (default: 10 => Recommended by original paper)
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(
        self,
        cfg: CFG,
        masking_budget: float = 0.15,
        span_probability: float = 0.2,
        max_span_length: int = 10
    ) -> None:
        super(SpanCollator, self).__init__(cfg)
        self.cfg = cfg
        self.tokenizer = self.cfg.tokenizer
        self.masking_budget = masking_budget
        self.span_probability = span_probability
        self.max_span_length = max_span_length

    def _whole_word_mask(
        self,
        input_tokens: List[str],
        max_predictions: int = CFG.max_seq
    ) -> List[int]:
        """
        0) apply Whole Word Masking Algorithm for make gathering original token index in natural language
        1) calculate number of convert into masking tokens with masking budget*len(input_tokens)
        2) define span length of this iteration
            - span length follow geometric distribution
            - span length is limited by max_span_length
        """
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if len(cand_indexes) >= 1 and self.select_post_string(token):  # method from WholeWordMaskingCollator
                cand_indexes[-1].append(i)
            elif self.select_src_string(token):  # method from WholeWordMaskingCollator
                cand_indexes.append([i])

        l = len(input_tokens)
        src_l = len(cand_indexes)
        num_convert_tokens = int(self.masking_budget * l)
        budget = num_convert_tokens  # int is immutable object, so do not copy manually
        masked_lms = []
        covered_indexes = set()
        while budget:
            span_length = max(1, min(10, int(torch.distributions.Geometric(probs=self.span_probability).sample())))
            src_index = random_non_negative_integer(src_l - 1)
            if span_length > budget:
                if budget < 5:  # Set the span length to budget to avoid a large number of iter if the remaining budget is too small
                    span_length = budget
                else:
                    continue
            if cand_indexes[src_index][0] + span_length > l - 1:  # If the index of the last token in the span is outside the full sequence range
                continue
            if len(cand_indexes[src_index]) > span_length:  # handling bad case: violating WWM algorithm at start
                continue
            span_token_index = cand_indexes[src_index][0]  # init span token index: src token
            while span_length:
                if span_length == 0:
                    break
                if span_token_index in covered_indexes: # If it encounters an index that is already masked, it ends, and starts the next iteration
                    break
                else:
                    covered_indexes.add(span_token_index)
                    masked_lms.append(span_token_index)
                    span_length -= 1
                    budget -= 1
                    span_token_index += 1
                    continue

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def get_mask_tokens(
        self,
        inputs: Tensor,
        mask_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """ Prepare masked tokens inputs/labels for Span Boundary Objective with MLM (15%),
        All of masked tokens (15%) are replaced by [MASK] token,
        Unlike BERT MLM which is replaced by random token or stay original token left
        """
        labels = inputs.clone()
        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ Abstract Method for Collator, you must implement this method in child class """
        input_ids = [torch.as_tensor(x["input_ids"]) for x in batched]
        padding_mask = [self.get_padding_mask(x) for x in input_ids]

        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        mask_labels = []
        for x in batched:
            ref_tokens = []
            for input_id in x["input_ids"]:
                token = self.tokenizer._convert_id_to_token(input_id)
                ref_tokens.append(token)
            mask_labels.append(self._whole_word_mask(ref_tokens))

        mask_labels = [torch.as_tensor(x) for x in mask_labels]
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=0)
        input_ids, labels = self.get_mask_tokens(
            input_ids,
            mask_labels
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
            "mask_labels": mask_labels
        }


class SBOHead(nn.Module):
    """ Custom Head for Span Boundary Objective Task, this module return logit value for each token
    we use z for class logit, each Fully Connected Layer doesn't have bias term in original paper
    so we don't use bias term in this module => nn.Linear(bias=False)

    You must select option for matrix sum or concatenate with x_s-1, x_e+1, p_i-s+1
    If you select concatenate option, you must pass is_concatenate=True to cfg.is_concatenate, default is True
    Math:
        h_0 = [x_s-1;x_e+1;p_i-s+1]
        h_t = LayerNorm(GELU(W_0•h_0))
        z = LayerNorm(GELU(W_1•h_t))

    Args:
        cfg: configuration.CFG
        is_concatenate: option for matrix sum or concatenate with x_s-1, x_e+1, p_i-s+1, default True
        max_span_length: maximum span length of each span in one batch sequence
                         (default: 10 => Recommended by original paper)
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(
        self,
        cfg: CFG,
        is_concatenate: bool = True,
        max_span_length: int = 10
    ) -> None:
        super(SBOHead, self).__init__()
        self.cfg = cfg
        self.is_concatenate = is_concatenate  # for matrix sum or concatenate with x_s-1, x_e+1, p_i-s+1
        self.projector = nn.Linear(self.cfg.dim_model, self.cfg.dim_model*3, bias=False)  # for concatenate x_s-1, x_e+1, p_i-s+1
        self.span_pos_emb = nn.Embedding(max_span_length*3, cfg.dim_model)  # size of dim_model is research topic
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim_model*3, self.cfg.dim_ffn),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_ffn),
            nn.Linear(self.cfg.dim_ffn, self.cfg.dim_model, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_model),
        )
        self.classifier = nn.Linear(self.cfg.dim_model, self.cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # for matching vocab size
        self.classifier.bias = self.bias

    @staticmethod
    def find_consecutive_groups(mask_labels: Tensor, target_value: int = 1) -> List[List[Dict]]:
        """ Get the start and end positions of consecutive groups in tensor for the target value
        This method is used for SBO Objective Function, this version is not best performance to make span groups

        Args:
            mask_labels: masking tensor for span
            target_value: target value for finding consecutive groups
        """
        all_consecutive_groups = []
        for mask_label in mask_labels:
            consecutive_groups = []
            current_group = None
            for i, value in enumerate(mask_label):
                if value == target_value:
                    if current_group is None:
                        current_group = {"start": i, "end": i}
                    else:
                        current_group["end"] = i
                else:
                    if current_group is not None:
                        consecutive_groups.append(current_group)
                        current_group = None
            if current_group is not None:
                consecutive_groups.append(current_group)
            all_consecutive_groups.append(consecutive_groups)
        return all_consecutive_groups

    def cal_span_emb(self, h: Tensor, hidden_states: Tensor, consecutive_groups: List[List[Dict]]) -> Tensor:
        """ Calculate span embedding for each span in one batch sequence

        Args:
            h: hidden states, already passed through projection layer (dim*3)
            hidden_states: hidden states from encoder
            consecutive_groups: consecutive groups for each batch sequence
        """
        for i, batch in enumerate(consecutive_groups):  # batch level
            for j, span in enumerate(batch):  # span level
                start, end = span["start"], span["end"]
                length = end - start + 1
                idx = torch.arange(length, device=self.cfg.device)   # .to(self.cfg.device)
                context_s, context_e = hidden_states[i, start - 1, :], hidden_states[i, end + 1, :]
                span_pos_emb = self.span_pos_emb(idx).squeeze(0)
                if length > 1:
                    for k, p_h in enumerate(span_pos_emb):  # length of span_pos_emb == length of span of this iterations
                        h[i, start+k, :] = torch.cat([context_s, p_h, context_e], dim=0)
                else:
                    h[i, start, :] = torch.cat([context_s, span_pos_emb, context_e], dim=0)
        return h

    def forward(self, hidden_states: Tensor, mask_labels: Tensor) -> Tensor:
        consecutive_groups = self.find_consecutive_groups(mask_labels)  # [batch, num_consecutive_groups]
        h = self.projector(hidden_states)  # [batch, seq, dim_model*3]
        h_t = self.cal_span_emb(h, hidden_states, consecutive_groups)
        z = self.head(h_t)
        logit = self.classifier(z)
        return logit
