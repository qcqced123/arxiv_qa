import re
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from configuration import CFG


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


class PretrainingMaskingCollator(nn.Module):
    """ Abstract Collator class for Pre-training with Masking (MLM, SBO ...) """
    def __init__(self):
        super(PretrainingMaskingCollator, self).__init__()
        self.pad_to_multiple_of = None

    def get_padding_mask(self, input_id: Tensor) -> Tensor:
        return torch.zeros(input_id.shape).bool()

    def get_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        raise NotImplementedError

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ Abstract Method for Collator, you must implement this method in child class """
        raise NotImplementedError


class SubWordMaskingCollator(PretrainingMaskingCollator):
    """ Custom Collator for Pretraining
    Masked Language Modeling Algorithm with Dynamic Masking from RoBERTa
        1) 15% of input tokens are selected at random for prediction
        2) 80% of the selected tokens are replaced with [MASK] token
        3) 10% of the selected tokens are replaced with random token in vocabulary
        4) The remaining 10% are left unchanged
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(SubWordMaskingCollator, self).__init__()
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer

    def get_mask_tokens(
        self,
        input_ids: Tensor,
        mlm_probability: float = 0.15,
        special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Tensor, Tensor]:
        """ Get Masked Tokens for MLM Task """
        input_ids = input_ids.clone()
        labels = input_ids.clone()

        # 1) 15% of input tokens are selected at random for prediction
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.as_tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 2) 80% of the selected tokens are replaced with [MASK] token
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        # 3) 10% of the selected tokens are replaced with random token in vocabulary
        random_words = torch.randint(
            self.tokenizer.vocab_size, labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ Masking for MLM with sub-word tokenizing """
        input_ids = [torch.as_tensor(x["input_ids"]) for x in batched]
        padding_mask = [self.get_padding_mask(x) for x in input_ids]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids, labels = self.get_mask_tokens(
            input_ids, self.cfg.mlm_probability
        )
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
        }


class WholeWordMaskingCollator(PretrainingMaskingCollator):
    """ Module for Whole Word Masking Task (WWM), basic concept is similar to MLM Task which is using sub-word tokenizer
    But, WWM do not allow sub-word tokenizing. Instead masking whole word-level token.
    In original source code, wwm is only applied to word-piece tokenizer in BERT Tokenizer,
    So, we extend original source code in Huggingface Transformers for applying wwm to bpe, bbpe, uni-gram tokenizer
    you must pass token, which is already normalized by tokenizer, to this module

    Example:
        1) sub-word mlm masking: pretrained => pre##, ##insert, ##ing => pre##, [MASK], ##ing
        2) whole-word mlm masking: pretrained => [MASK], [MASK], [MASK]

    extend:
        original source code:
            if len(cand_indexes) >= 1 and token.startswith("##"):
        extended source code:
            use flag value with method select_string()

    References:
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG) -> None:
        super(WholeWordMaskingCollator, self).__init__()
        self.cfg = cfg
        self.mlm_probability = cfg.mlm_probability
        self.tokenizer = cfg.tokenizer

        if self.tokenizer.__class__.__name__ in SPM:
            self.tokenizer_type = 'SPM'
        elif self.tokenizer.__class__.__name__ in BPE:
            self.tokenizer_type = 'BPE'
        elif self.tokenizer.__class__.__name__ in WORDPIECE:
            self.tokenizer_type = 'WORDPIECE'

    def select_src_string(self, token: str) -> bool:
        """ set flag value for selecting src tokens to mask in sub-word
        Args:
            token: str, token to check
        """
        flag = False
        if self.tokenizer_type == 'SPM':
            flag = True if token.startswith("▁") else False

        elif self.tokenizer_type == 'BPE':
            flag = True if token.startswith("Ġ") else False

        elif self.tokenizer_type == 'WORDPIECE':
            pattern = re.compile(r'[^\w\d\s]|_')
            flag = False if re.match(pattern, token) else True
        return flag

    def select_post_string(self, token: str) -> bool:
        """ set flag value for selecting post tokens to mask in sub-word
        Args:
            token: str, token to check
        """
        flag = False
        if self.tokenizer_type == 'SPM':
            pattern = re.compile(r'[^\w\d\s]|_')
            flag = False if re.match(pattern, token[0]) else True

        elif self.tokenizer_type == 'BPE':
            flag = False if token.startswith("Ġ") else True

        elif self.tokenizer_type == 'WORDPIECE':
            flag = True if token.startswith("##") else False

        return flag

    def _whole_word_mask(
            self,
            input_tokens: List[str],
            max_predictions: int = CFG.max_seq
    ) -> List[int]:
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if len(cand_indexes) >= 1 and self.select_post_string(token):
                cand_indexes[-1].append(i)
            elif self.select_src_string(token):
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def get_mask_tokens(
        self,
        inputs: Tensor,
        mask_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling(15%):
        80% MASK, 10% random, 10% original. Set 'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref
        """
        labels = inputs.clone()
        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.as_tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ Masking for MLM with whole-word tokenizing """
        input_ids = [x["input_ids"] for x in batched]
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
        inputs, labels = self.get_mask_tokens(
            input_ids,
            mask_labels
        )
        return {
            "input_ids": inputs,
            "labels": labels,
            "padding_mask": padding_mask,
        }


class MLMCollator:
    """ Custom Collator for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    Dataloader returns a list to collator, so collator should be able to handle list of tensors
    Args:
        cfg: configuration.CFG
        is_mlm: whether to use MLM or not, if you set False to this argument,
                this collator will be used for CLM
        special_tokens_mask: special tokens mask for masking
    References:
        https://huggingface.co/docs/transformers/v4.32.0/ko/tasks/masked_language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG, is_mlm: bool = True, special_tokens_mask: Optional[Any] = None):
        self.cfg = cfg
        self.mlm = is_mlm
        self.special_tokens_mask = special_tokens_mask

    def __call__(self, batched: List[Dict[str, Tensor]]) -> Dict:
        batch_instance = None
        if self.cfg.mlm_masking == 'SubWordMasking':
            batch_instance = SubWordMaskingCollator(self.cfg)(batched)
        if self.cfg.mlm_masking == 'WholeWordMasking':
            batch_instance = WholeWordMaskingCollator(self.cfg)(batched)
        return batch_instance


class MLMHead(nn.Module):
    """
    Custom Masked Language Model Head for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    For Encoder, Such as BERT, RoBERTa, ELECTRA, DeBERTa, ... etc
    Args:
        cfg: configuration.CFG
    Notes:
        class var named "decoder" means denoise masking token for prediction, not Transformer Decoder
    References:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L681
    """
    def __init__(self, cfg: CFG) -> None:
        super(MLMHead, self).__init__()
        self.fc = nn.Linear(cfg.dim_model, cfg.dim_model)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.dim_model, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # for matching vocab size
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.fc(hidden_states)
        x = self.gelu(x)
        ln_x = self.layer_norm(x)
        logit = self.decoder(ln_x)
        return logit



