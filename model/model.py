import torch
import torch.nn as nn
import model.pooling as pooling

from torch import Tensor
from typing import List, Dict, Tuple


from model.mlm import MLMHead
from model.clm import CLMHead
from configuration import CFG
from trainer.loss import ArcFace
from model.abstract_task import AbstractTask
from model.model_utils import freeze, reinit_topk


class MaskedLanguageModel(nn.Module, AbstractTask):
    """ Custom Model for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    You can use backbone model as BERT, DeBERTa, Linear Transformer, Roformer ...

    Args:
        cfg: configuration.CFG

    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG) -> None:
        super(MaskedLanguageModel, self).__init__()
        self.cfg = cfg
        self.model = self.select_model(cfg.num_layers)
        self.mlm_head = MLMHead(cfg)

        self._init_weights(self.model)
        self._init_weights(self.mlm_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        outputs = self.model(inputs, padding_mask)
        return outputs

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> List[Tensor]:
        last_hidden_states, _ = self.feature(inputs, padding_mask)
        logit = self.mlm_head(last_hidden_states)
        return logit


class CasualLanguageModel(nn.Module, AbstractTask):
    """ Custom Model for CLM Task, which is used for pre-training Auto-Regressive Model (AR)

    Also, you can use this task module for text generation,
    text summarization with generation ... any other task ,which is needed to generate text from prompt sentences

    Notes:
        L = L_CLM (pure language modeling)

    Args:
        cfg: configuration.CFG

    References:
        https://huggingface.co/docs/transformers/main/tasks/language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG) -> None:
        super(CasualLanguageModel, self).__init__()
        self.cfg = cfg

        # select model from local non-trained model or pretrained-model from huggingface hub
        if self.cfg.use_pretrained:
            self.components = self.select_pt_model(generate_mode=self.cfg.generate_mode)
            self.auto_cfg = self.components['plm_config']
            self.model = self.components['plm']
            self.lm_head = CLMHead(cfg, self.auto_cfg) if not self.cfg.generate_mode else None  # for generation mode

        else:
            self.model = self.select_model(cfg.num_layers)
            self.lm_head = CLMHead(cfg)
            self._init_weights(self.model)

        if self.lm_head is not None:
            self._init_weights(self.lm_head)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(self, inputs: Dict) -> Tensor:
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        return outputs

    def forward(self, inputs: Dict) -> List[Tensor]:
        last_hidden_states, logit = None, None
        if self.cfg.use_pretrained:
            if self.cfg.generate_mode:  # for generation mode or pretraining mode with AutoModelForCausalLM
                _, logit, _, last_hidden_states, _ = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

            else:  # for pretraining mode with AutoModel
                last_hidden_states = self.feature(inputs).last_hidden_state
        else:
            last_hidden_states, _ = self.feature(inputs)
        logit = self.lm_head(last_hidden_states) if self.lm_head is not None else logit
        return logit


class SentimentAnalysis(nn.Module, AbstractTask):
    """ Fine-Tune Task Module for Sentiment Analysis Task, same as multi-class classification task, not regression tasks
    We set target classes as 5, which is meaning of 1 to 5 stars

    All of dataset should be unified by name rule, for making prompt sentences and labels range 1 to 5 stars rating

        1) if your dataset's column name's are not unified
            - please add new keys to name_dict in dataset_class/name_rule/sentiment_analysis.py

        2) if your dataset's target labels are not range 1 to 5
            - ASAP, We make normalizing function for target labels range 1 to 5 rating
    """
    def __init__(self, cfg: CFG) -> None:
        super(SentimentAnalysis, self).__init__()
        self.cfg = cfg
        self.components = self.select_pt_model()
        self.model = self.components['plm']
        self.auto_cfg = self.components['plm_config']
        self.prompt_encoder = self.components['prompt_encoder']

        # self.model.resize_token_embeddings(len(self.cfg.tokenizer))

        self.pooling = getattr(pooling, self.cfg.pooling)(self.cfg)
        self.fc = nn.Linear(
            self.auto_cfg.hidden_size,
            self.cfg.num_labels,
            bias=False
        )

        self._init_weights(self.fc)
        if self.cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:self.cfg.num_freeze])

        if self.cfg.reinit:
            reinit_topk(self.model, self.cfg.num_reinit)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(self, inputs: Dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: Dict) -> Tensor:
        """ need to implement p-tuning options in forward function
        """
        h = self.feature(inputs)
        features = h.last_hidden_state

        if self.cfg.pooling == 'WeightedLayerPooling':  # using all encoder layer's output
            features = h.hidden_states

        embedding = self.pooling(features, inputs['attention_mask'])
        logit = self.fc(embedding)
        return logit


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
    def __init__(self, cfg: CFG, len_train: int) -> None:
        super(MetricLearningModel, self).__init__()
        self.cfg = cfg
        self.components = self.select_pt_model()
        self.auto_cfg = self.components['plm_config']
        self.model = self.components['plm']
        self.prompt_encoder = self.components['prompt_encoder']

        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        self.pooling = getattr(pooling, self.cfg.pooling)(self.cfg)
        self.arcface_head = ArcFace(
            dim_model=self.auto_cfg.hidden_size,
            num_classes=len_train,
            device=self.cfg.device
        )  # parameter sharing, called in trainer class's train method, not in this method

        if self.cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:self.cfg.num_freeze])

        if self.cfg.reinit:
            reinit_topk(self.model, self.cfg.num_reinit)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def get_mask(self, features: Tensor, mask_type: str, query_index: Tensor, context_index: Tensor = None) -> Tensor:
        """ return the mask for separating the query and context sentence embedding """
        mask = None
        batch_size, seq_length, hidden_size = features.size()
        if mask_type == 'query':
            mask = (torch.arange(seq_length).expand(batch_size, seq_length).to(self.cfg.device) >= 1) & (torch.arange(seq_length).expand(batch_size, seq_length).to(self.cfg.device) < query_index.unsqueeze(1))
        elif mask_type == 'context':
            mask = (torch.arange(seq_length).expand(batch_size, seq_length).to(self.cfg.device) >= query_index.unsqueeze(1) + 1) & (torch.arange(seq_length).expand(batch_size, seq_length) < context_index.unsqueeze(1))
        return mask

    def feature(self, inputs: Dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: Dict, query_index: Tensor, context_index: Tensor) -> Tuple[Tensor, Tensor]:
        """ forward function for Metric Learning Task, which is used for learning similarity between two input sentences
        using argument "query_index" and "context_index" for separating two input sentences

        Args:
            inputs: Dict, input dictionary for forward function
            query_index: int, index of query token in input_ids
            context_index: int, index of context token in input_ids

        Returns:
            query_h: Tensor, embedding of query token
            context_h: Tensor, embedding of context token
        """
        h = self.feature(inputs)
        features = h.last_hidden_state

        print(f"query_index is: {query_index}")
        print(f"context_index is: {context_index}")

        print(f"query_index shape is: {query_index.shape}")
        print(f"context_index shape is: {context_index.shape}")

        if self.cfg.pooling == 'WeightedLayerPooling':  # using all encoder layer's output
            features = h.hidden_states

        # for calculating the arcface loss, multiple negative ranking loss
        query_h = self.pooling(
            last_hidden_state=features,
            mask=self.get_mask(features=features, mask_type='query', query_index=query_index),
            p=self.cfg.pow_value
        )
        # for calculating the arcface loss, multiple negative ranking loss
        context_h = self.pooling(
            last_hidden_state=features,
            mask=self.get_mask(features, mask_type='context', query_index=query_index, context_index=context_index),
            p=self.cfg.pow_value
        )
        return query_h, context_h



