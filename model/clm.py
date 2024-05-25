import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List
from .mlm import WholeWordMaskingCollator
from configuration import CFG


class CLMCollator(WholeWordMaskingCollator):
    def __init__(self, cfg: CFG) -> None:
        super(CLMCollator, self).__init__(cfg=cfg)

    def get_mask_tokens(self, inputs: Tensor, pad_mask: Tensor) -> Tensor:
        """ make masking matrix for Decoder Masked Multi-Head Self-attention
        combine padding mask and attention mask, naming as attention_mask

        Notes:
            current lm_mask's dtype is torch.bool, should be careful when you set true in mixed precision training options
        """
        lm_mask = torch.triu(
            torch.ones(inputs.shape[0], inputs.shape[-1], inputs.shape[-1], dtype=torch.bool),
            diagonal=1
        )  # [batch, seq_len, seq_len]
        pad_mask = pad_mask.unsqueeze(1).expand(-1, pad_mask.shape[1], -1)  # [batch, seq_len, seq_len]
        attention_mask = lm_mask | pad_mask
        return attention_mask

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ masking for CLM Task
        return:
            input_ids: padded input_ids
            labels: rolled input_ids, blocking last token predictions
            attention_mask: padded attention_mask
        """
        input_ids = [x["input_ids"] for x in batched]
        padding_mask = [self.get_padding_mask(x) for x in input_ids]

        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = input_ids.clone().roll(shifts=-1, dims=-1)
        labels[:, -1] = -100  # for blocking predicting last token such as SEP, EOS

        attention_mask = self.get_mask_tokens(input_ids, padding_mask) if not self.cfg.use_pretrained else pad_sequence([x["attention_mask"] for x in batched], batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class CLMHead(nn.Module):
    """ Custom Casual Language Model Head for CLM Task, which is used for pre-training Auto-Regressive Model (AR)
    CLM decoder does not use bias term, so set nn.Linear bias options as False
    For Decoder, Such as GPT2, GPTNeo, ...

    Args:
        cfg: configuration.CFG

    References:
        https://huggingface.co/docs/transformers/main/tasks/language_modeling.html
    """

    def __init__(self, cfg: CFG, pretrained_cfg=None) -> None:
        super(CLMHead, self).__init__()
        self.cfg = cfg
        self.dim_model = cfg.dim_model if not self.cfg.use_pretrained else pretrained_cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.decoder = nn.Linear(self.dim_model, self.vocab_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states
        logit = self.decoder(x)
        return logit


