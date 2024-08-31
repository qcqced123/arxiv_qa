import torch.nn as nn
import model.pooling as pooling

from torch import Tensor
from configuration import CFG
from typing import Dict, Tuple
from model.abstract_task import AbstractTask
from model.model_utils import freeze, reinit_topk


class MetricLearningModel(nn.Module, AbstractTask):
    """ custom model for Metric Learning Task, which is used for learning similarity between two input sentences
    We use this model for learning similarity between two input sentences, which is used for semantic search

    we set the arcface & multiple negative ranking loss combined pipeline (multi-objective learning)
    you can see total structure of this pipeline in below references link

    we use arcface head(weight shared) and batch multiple negative ranking loss,
    originally, we must add the two arcface head for each type of sentence, but we use shared weight for two sentences
    it can be available because of the same number of instances in each type of sentence, expecting for reducing the number of parameters and training time

    Args:
        cfg: configuration.CFG

    References:
        https://arxiv.org/pdf/2005.11401
        https://arxiv.org/abs/1801.07698
        https://arxiv.org/pdf/1705.00652.pdf
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://www.sbert.net/docs/package_reference/losses.html
        https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr/notebook
        https://www.youtube.com/watch?v=b_2v9Hpfnbw&ab_channel=NicholasBroad
    """
    def __init__(self, cfg: CFG) -> None:
        super(MetricLearningModel, self).__init__()
        self.cfg = cfg
        self.components = self.select_pt_model(mode="text-similarity")

        self.model = self.components['plm']
        self.auto_cfg = self.components['plm_config']
        self.prompt_encoder = self.components['prompt_encoder']
        self.pooling = getattr(pooling, self.cfg.pooling)(self.cfg)

        if self.cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:self.cfg.num_freeze])

        if self.cfg.reinit:
            reinit_topk(self.model, self.cfg.num_reinit)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, inputs: Dict, query_mask: Tensor, context_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """ forward function for Metric Learning Task, which is used for learning similarity between two input sentences
        using argument "query_index" and "context_index" for separating two input sentences

        Args:
            inputs: Dict, input dictionary for forward function
            query_mask: torch.Tensor, boolean mask for query sentence
            context_mask: torch.Tensor, boolean mask for context(document) sentence

        Returns:
            query_h: Tensor, embedding of query token
            context_h: Tensor, embedding of context token
        """
        h = self.model(**inputs)
        features = h.last_hidden_state

        # for calculating the batch infoNCE
        query_h = self.pooling(
            last_hidden_state=features,
            mask=query_mask,
            p=self.cfg.pow_value
        )
        context_h = self.pooling(
            last_hidden_state=features,
            mask=context_mask,
            p=self.cfg.pow_value
        )
        return query_h, context_h


