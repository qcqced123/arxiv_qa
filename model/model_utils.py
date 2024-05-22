import torch.nn as nn
from typing import List


def freeze(module: nn.Module) -> None:
    """ Freezes module's parameters
    Args:
        module: target module to freeze

    Examples:
        freezing embeddings and first 2 layers of encoder
        1) freeze(model.embeddings
        2) freeze(model.encoder.layer[:2])
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freeze_parameters(module: nn.Module) -> List[str]:
    """ Returns names of freeze parameters of the given module.

    Args:
        module: target module to get freeze parameters

    Examples:
        freeze_parameters = get_freeze_parameters(model)
    """
    freeze_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freeze_parameters.append(name)
    return freeze_parameters


def reinit_topk(model: nn.Module, num_layers: int) -> None:
    """ Re-initialize the last-k transformer Encoder layers.
    Encoder Layer: Embedding, Attention Head, LayerNorm, Feed Forward
    Initialization will follow the default initialization of the model, which is setting in config json file

    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers > 0:
        model.encoder.layer[-num_layers:].apply(model._init_weights)
