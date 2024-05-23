import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelectDistances(nn.Module):
    """
    Select Distance Metrics
    Args:
        metrics: select distance metrics do you want,
          - options: euclidean, manhattan, cosine
          - PairwiseDistance: Manhattan(p=1), Euclidean(p=2)
          - type: you must pass str type
    """
    def __init__(self, metrics: str) -> None:
        super().__init__()
        self.metrics = metrics

    @staticmethod
    def sim_matrix(a: Tensor, b: Tensor, eps=1e-8) -> Tensor:
        """ added eps for numerical stability """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def select_distance(self, x: Tensor, y: Tensor) -> Tensor:
        if self.metrics == 'cosine':
            distance_metric = 1 - self.sim_matrix(x, y)  # Cosine Distance
        elif self.metrics == 'euclidean':
            distance_metric = F.pairwise_distance(x, y, p=2)  # L2, Euclidean
        else:
            distance_metric = F.pairwise_distance(x, y, p=1)  # L1, Manhattan
        return distance_metric


