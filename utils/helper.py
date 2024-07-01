import torch, os, sys, random, json
import numpy as np
from pathlib import Path
from typing import List


def calculate_layerwise_lr(max_lr: float, num_layers: int, lr_decay: float) -> List[float]:
    """ function for calculating the layerwise learning rate
    this function return the lr for each layer in the model, which is decreasing by the lr_decay rate

    Args:
        max_lr: float, maximum learning rate for the model, lr at the top(last) of the given models
        num_layers: int, number of layers in the model
        lr_decay: float, decay rate for the learning rate, this value means that decay rate of the learning rate per layer

    Returns:
        List[float]: list of learning rates for each layer in the model

    """
    return [max_lr * (lr_decay ** i) for i in range(num_layers)][::-1]


def select_model_file(base_path: str = 'models', arch: str = None, model: str = None) -> str:
    """ Construct the full path using Path() """
    path = base_path + '/' + arch + '/' + model
    full_path = Path(path)
    if full_path.exists():
        return str(full_path)
    else:
        raise FileNotFoundError(f"The file {model} does not exist in the specified path.")


def check_device() -> bool:
    return torch.mps.is_available()


def check_library(checker: bool) -> tuple:
    """
    1) checker == True
        - current device is mps
    2) checker == False
        - current device is cuda with cudnn
    """
    if not checker:
        _is_built = torch.backends.cudnn.is_available()
        _is_enable = torch.backends.cudnn.enabled
        version = torch.backends.cudnn.version()
        device = (_is_built, _is_enable, version)
        return device


def class2dict(cfg) -> dict:
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


def all_type_seed(cfg, checker: bool) -> None:
    # python & torch seed
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)  # python Seed
    random.seed(cfg.seed)  # random module Seed
    np.random.seed(cfg.seed)  # numpy module Seed
    torch.manual_seed(cfg.seed)  # Pytorch CPU Random Seed Maker

    # device == cuda
    if not checker:
        torch.cuda.manual_seed(cfg.seed)  # Pytorch GPU Random Seed Maker
        torch.cuda.manual_seed_all(cfg.seed)  # Pytorch Multi Core GPU Random Seed Maker
        # torch.cudnn seed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # devide == mps
    else:
        torch.mps.manual_seed(cfg.seed)


def seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


