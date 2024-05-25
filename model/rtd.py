import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from configuration import CFG


def get_discriminator_input(inputs: Tensor, labels: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
    """ Post Processing for Replaced Token Detection Task
    1) get index of the highest probability of [MASK] token in pred tensor
    2) convert [MASK] token to prediction token
    3) make label for Discriminator

    Args:
        inputs: pure inputs from tokenizing by tokenizer
        labels: labels for masked language modeling
        pred: prediction tensor from Generator

    returns:
        d_inputs: torch.Tensor, shape of [Batch, Sequence], for Discriminator inputs
        d_labels: torch.Tensor, shape of [Sequence], for Discriminator labels
    """
    # 1) flatten pred to 2D Tensor
    d_inputs, d_labels = inputs.clone().detach().view(-1), None  # detach to prevent back-propagation
    flat_pred, flat_label = pred.view(-1, pred.size(-1)), labels.view(-1)  # (batch * sequence, vocab_size)

    # 2) get index of the highest probability of [MASK] token
    pred_token_idx, mlm_mask_idx = flat_pred.argmax(dim=-1), torch.where(flat_label != -100)
    pred_tokens = torch.index_select(pred_token_idx, 0, mlm_mask_idx[0])

    # 3) convert [MASK] token to prediction token
    d_inputs[mlm_mask_idx[0]] = pred_tokens

    # 4) make label for Discriminator
    original_tokens = inputs.clone().detach().view(-1)
    original_tokens[mlm_mask_idx[0]] = flat_label[mlm_mask_idx[0]]
    d_labels = torch.eq(original_tokens, d_inputs).long()  # unnecessary recover to original label shape, keep flatten state
    d_inputs = d_inputs.view(pred.size(0), -1)  # covert to [batch, sequence]
    return d_inputs, d_labels


class RTDCollator(nn.Module):
    """ Replaced Token Detection Collator (RTD) for Pretraining
    from ELECTRA original paper
    """
    pass


class RTDHead(nn.Module):
    """ Replaced Token Detection Head (RTD) for Pretraining from ELECTRA original paper
    RTD Task is same as Binary Classification Task (BCE in pytorch)
    classes: 0 (replaced) or 1 (original)
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(RTDHead, self).__init__()
        self.cfg = cfg
        self.classifier = nn.Linear(self.cfg.dim_model, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        logit = self.classifier(hidden_states)
        return logit
