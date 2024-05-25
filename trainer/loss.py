import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Dict
from trainer.distance_metric import SelectDistances


class SmoothL1Loss(nn.Module):
    """ Smooth L1 Loss in Pytorch """
    def __init__(self, reduction: str = 'mean') -> None:
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.SmoothL1Loss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class MSELoss(nn.Module):
    """ Mean Squared Error Loss in Pytorch """
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.MSELoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class RMSELoss(nn.Module):
    """ Root Mean Squared Error Loss in Pytorch
    Args:
        reduction: str, reduction method of losses
        eps: float, epsilon value for numerical stability (Defending Underflow, Zero Division)
    """
    def __init__(self, reduction: str = 'mean', eps=1e-8) -> None:
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps  # If MSE == 0, We need eps

    def forward(self, yhat, y) -> Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    """
    Mean Column-wise Root Mean Squared Error Loss
    Calculate RMSE per target values(columns), and then calculate mean of each column's RMSE Result
    Args:
        reduction: str, reduction method of losses
        num_scored: int, number of scored target values, default 1 same as RMSELoss
    """
    def __init__(self, reduction: str, num_scored: int = 1) -> None:
        super(MCRMSELoss, self).__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + (self.RMSE(yhat[:, i], y[:, i]) / self.num_scored)
        return score


class WeightMCRMSELoss(nn.Module):
    """
    Apply losses rate per target classes
    Weighted Loss can transfer original label data's distribution to pseudo label data
    References:
        https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609
    """
    def __init__(self, reduction, num_scored=6):
        super(WeightMCRMSELoss, self).__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored
        self._loss_rate = torch.tensor([0.21, 0.16, 0.10, 0.16, 0.21, 0.16], dtype=torch.float32)

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + torch.mul(self.RMSE(yhat[:, i], y[:, i]), self._loss_rate[i])
        return score


class WeightedMSELoss(nn.Module):
    """ Weighted MSE Loss
    Reference:
        https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369793
    """
    def __init__(self, reduction, task_num=1) -> None:
        super(WeightedMSELoss, self).__init__()
        self.task_num = task_num
        self.smoothl1loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, y_pred, y_true, log_vars) -> float:
        loss = 0
        for i in range(self.task_num):
            precision = torch.exp(-log_vars[i])
            diff = self.smoothl1loss(y_pred[:, i], y_true[:, i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        loss = 0.5*loss
        return loss


class KLDivLoss(nn.Module):
    """ KL-Divergence Loss
    Args:
        reduction: str, reduction method of losses
    """
    def __init__(self, reduction: str = 'batchmean') -> None:
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        criterion = nn.KLDivLoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class BinaryCrossEntropyLoss(nn.Module):
    """ Binary Cross-Entropy Loss for Binary Classification
    Args:
        reduction: str, reduction method of losses
    """
    def __init__(self, reduction):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class CrossEntropyLoss(nn.Module):
    """ Cross-Entropy Loss for Multi-Class Classification """
    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        criterion = nn.CrossEntropyLoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class PearsonLoss(nn.Module):
    """ Pearson Correlation Coefficient Loss """
    def __init__(self, reduction: str = 'mean') -> None:
        super(PearsonLoss, self).__init__()
        self.reduction = reduction

    @staticmethod
    def forward(y_pred, y_true) -> Tensor:
        x = y_pred.clone()
        y = y_true.clone()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr = torch.maximum(torch.minimum(corr, torch.tensor(1)), torch.tensor(-1))
        return torch.sub(torch.tensor(1), corr ** 2)


class CosineEmbeddingLoss(nn.Module):
    """ Cosine Embedding Loss for Metric Learning, same concept with Contrastive Loss
    but some setting is different, Cosine Embedding Loss is more range to [-1, 1]

    This Module is API Wrapper of nn.CosineEmbeddingLoss from pytorch
    Args:
        reduction: str, reduction method of losses
        margin: float, default = 0, margin value for cosine embedding loss
    """
    def __init__(self, reduction: str = 'mean', margin: float = 0.0) -> None:
        super(CosineEmbeddingLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin

    def forward(self, y_pred: Tensor, y_true: Tensor, label: Tensor) -> Tensor:
        criterion = nn.CosineEmbeddingLoss(margin=self.margin, reduction=self.reduction)
        return criterion(
            y_pred,
            y_true,
            label
        )


# Contrastive Loss for NLP Semantic Search
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss which is basic method of Metric Learning
    Closer distance between data points in intra-class, more longer distance between data points in inter-class
    Distance:
        Euclidean Distance: sqrt(sum((x1 - x2)**2))
        Cosine Distance: 1 - torch.nn.function.cos_sim(x1, x2)
    Examples:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)
    Args:
        margin: margin value meaning for Area of intra class(positive area), default 1.0
        metric: standard of distance metrics, default cosine distance
    References:
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
        https://www.youtube.com/watch?v=u-X_nZRsn5M&list=LL&index=3&t=10s&ab_channel=DeepFindr
    """
    def __init__(self, metric: str = 'cosine', margin: int = 1.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.distance = SelectDistances(metric)  # Options: euclidean, manhattan, cosine
        self.margin = margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        anchor, instance = embeddings
        distance = self.distance(anchor, instance)
        contrastive_loss = 0.5 * (labels.float() * torch.pow(distance, 2) +
                                  (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return contrastive_loss.mean()


# Batch Embedding Table Dot-Product Version of Contrastive Loss
class BatchDotProductContrastiveLoss(nn.Module):
    """
    Batch Embedding Inner-Product based Contrastive Loss Function
    This metrics request just one input batch, not two input batches for calculating losses
    Args:
        metric: distance metrics, default: cosine
        margin: margin for negative pairs, default: 1.0
    Maths:
        losses(x, y) = 0.5 * (y * distance(x1, x2) + (1 - y) * max(margin - distance(x1, x2), 0))
    """
    def __init__(self, metric: str = 'cosine', margin: int = 1.0) -> None:
        super(BatchDotProductContrastiveLoss, self).__init__()
        self.distance = SelectDistances(metric)  # Options: euclidean, manhattan, cosine
        self.margin = margin

    @staticmethod
    def elementwise_labels(y: torch.Tensor) -> torch.Tensor:
        """ Element-wise masking matrix generation for contrastive losses by label """
        f_mask = (y == 1) | (y == -1)
        t_mask = (y == 2) | (y == 0)
        y = y.unsqueeze(1)
        label = y + y.T
        label[f_mask], label[t_mask] = -1, 1
        label[f_mask] = 0
        label = torch.triu(label, diagonal=1)
        return label

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = self.elementwise_labels(labels)
        distance = self.distance.select_distance(emb, emb)
        contrastive_loss = 0.5 * (labels.float() * torch.pow(distance, 2) +
                                  (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        contrastive_loss = torch.sum(torch.triu(contrastive_loss, diagonal=1))
        return contrastive_loss / labels.shape[0]


class ArcFace(nn.Module):
    """
    ArcFace Pytorch Implementation
    Args:
        dim_model: size of hidden states(latent vector)
        num_classes: num of target classes
        s: re-scale scaler, default 30.0
        m: Additive Angular Margin Penalty
    References:
        https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py
        https://arxiv.org/abs/1801.07698
    """
    def __init__(self, dim_model: int, num_classes: int, s: int = 30.0, m: int = 0.50) -> None:
        super(ArcFace, self).__init__()
        self.dim_model = dim_model
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.FloatTensor(num_classes, dim_model))
        nn.init.kaiming_uniform_(self.w)  # same as nn.Linear

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - m)  # for taylor series
        self.mm = math.sin(math.pi - m) * m  # for taylor series

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        cosine = torch.matmul(F.normalize(self.w), F.normalize(inputs))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        z = cosine*self.cos_m - sine*self.sin_m
        z = torch.where(cosine > self.th, z, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        loss = (one_hot * z) + ((1.0 - one_hot) * cosine)
        loss *= self.s
        return loss

