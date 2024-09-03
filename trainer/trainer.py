import gc
import wandb
import transformers

import torch
import torch.nn as nn
import trainer.loss as loss
import trainer.metric as metric
import dataset_class.dataclass as dataset_class

from tqdm.auto import tqdm
from configuration import CFG
from model import model as task
from elasticsearch import Elasticsearch
from typing import Union, List, Callable, Dict
from transformers import AutoConfig, AutoTokenizer

from db.run_db import get_encoder
from dataset_class.preprocessing import dataset_split, load_all_types_dataset

from trainer.collator_fn import BatchCollatorFunc
from trainer.trainer_utils import AverageMeter, get_dataloader
from trainer.trainer_utils import get_optimizer_grouped_parameters, get_scheduler


class MetricLearningTuner:
    """ trainer class for conducting metric learning objective, such as contrastive learning,
    multiple negative ranking loss, arcface, infoNCE ...

    this trainer class is designed for especially getting good retriever between user's query and document

    so, we use the plm for the retrieval task such as sentence-transformers, DPR

    also, we use the plm such as longformer, bigbird for the retrieval task, which can have only encoder and
    can be good at long sequence (more than 512 tokens)

    we use arcface head(weight shared) and batch multiple negative ranking loss,
    originally, we must add the two arcface head for each type of sentence, but we use shared weight for two sentences
    it can be available because of the same number of instances in each type of sentence, expecting for reducing the number of parameters and training time

    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed

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
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.generator = generator
        self.metric_list = self.cfg.metrics
        self.plm_cfg = AutoConfig.from_pretrained(self.cfg.model_name)
        self.df = load_all_types_dataset(f'./dataset_class/datafolder/arxiv_qa/arxiv_question_document_pair.csv')

    @staticmethod
    def save_model(model: nn.Module, config: AutoConfig, tokenizer: AutoTokenizer, to: str) -> None:
        """ save the current state of model's config, tokenizer config

        Args:
            model (nn.Module):
            config (AutoConfig):
            tokenizer (AutoTokenizer):
            to (str): save path for model, config, tokenizer

        Returns:
            None
        """
        print(f"Saving {model.dtype} model to {to}...")
        model.save_pretrained(to)
        tokenizer.save_pretrained(to)
        config.save_pretrained(to)

    def make_batch(self) -> Union:
        train, valid = dataset_split(self.cfg, self.df)
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid)

        collate_fn = BatchCollatorFunc(plm_cfg=self.plm_cfg)
        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            generator=self.generator,
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            batch_size=self.cfg.val_batch_size,
            collate_fn=collate_fn,
            generator=self.generator,
            shuffle=False,
            pin_memory=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(self.df)

    def model_setting(self, len_train: int) -> Union:
        # initialize and get the model module
        # transfer model in CPU to GPU
        model = getattr(task, self.cfg.task)(self.cfg)
        model.to(self.cfg.device)

        # initialize and get the loss, metric module
        criterion = getattr(loss, self.cfg.loss_fn)()
        val_criterion = getattr(loss, self.cfg.val_loss_fn)()
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

        # set up the weight updating method
        # option: backward with layer-wise learning rate decay, pure backward
        if self.cfg.llrd:
            lr = self.cfg.layerwise_lr
            optimizer_params = get_optimizer_grouped_parameters(
                model,
                lr,
                self.cfg.weight_decay,
                self.cfg.layerwise_lr_decay
            )
        else:
            lr = self.cfg.lr
            optimizer_params = model.parameters()

        # get optimizer module
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=optimizer_params,
            lr=lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
        )

        # get scheduler module
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)
        return model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler

    def train_val_fn(
        self,
        loader_train,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        loader_valid,
        val_criterion: nn.Module,
        val_metric_list: List[Callable],
        val_score_max: float,
    ) -> Union:

        model.train()
        losses = AverageMeter()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)

        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad(set_to_none=True)
            inputs = {
                k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
            }
            query_mask = batch['query_mask'].to(self.cfg.device, non_blocking=True)
            document_mask = batch['document_mask'].to(self.cfg.device, non_blocking=True)

            batch_size = query_mask.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                query_h, context_h = model(
                    inputs=inputs,
                    query_mask=query_mask,
                    context_mask=document_mask
                )
                loss = criterion(query_h, context_h)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.item(), batch_size)

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # logging insert loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.item()

            wandb.log({
                '<Per Step> Total Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # branch for conducting validation stage when the current step value is same as config's "val_check" value
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                gc.collect()  # avoiding the kernel shut-down caused by memory
                valid_loss = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                if val_score_max >= valid_loss:
                    print(f'[Update] Total Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Total Best Score: {valid_loss}')
                    self.save_model(
                        model=model.model,
                        config=model.auto_cfg,
                        tokenizer=self.cfg.tokenizer,
                        to=self.cfg.checkpoint_dir
                    )
                    val_score_max = valid_loss

        gc.collect()
        torch.cuda.empty_cache()

        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def valid_fn(
        self,
        loader_valid,
        model: nn.Module,
        val_criterion: nn.Module,
        val_metric_list: List[Callable]
    ) -> float:
        model.eval()
        valid_losses = AverageMeter()
        valid_metrics = {
            "positive_cosine_similarity": AverageMeter(),
            "negative_cosine_similarity": AverageMeter(),
        }
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = {
                    k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
                }
                query_mask = batch['query_mask'].to(self.cfg.device, non_blocking=True)
                document_mask = batch['document_mask'].to(self.cfg.device, non_blocking=True)

                batch_size = document_mask.size(0)
                query_h, context_h = model(
                    inputs=inputs,
                    query_mask=query_mask,
                    context_mask=document_mask
                )
                loss = val_criterion(query_h, context_h)
                valid_losses.update(loss.item(), batch_size)

                wandb.log({
                    '<Val Step> valid loss': valid_losses.avg
                })

                # calculate the cosine similarity between positive pair of query and document
                queries, contexts = query_h.detach().cpu(), context_h.detach().cpu()
                for i, metric_fn in enumerate(val_metric_list):
                    pos_score, neg_score = metric_fn(
                        queries,
                        contexts,
                    )
                    valid_metrics["positive_cosine_similarity"].update(pos_score, batch_size)
                    valid_metrics["negative_cosine_similarity"].update(neg_score, batch_size)

                    wandb.log({
                        f'<Val Step> pos cosine sim': valid_metrics["positive_cosine_similarity"].avg,
                        f'<Val Step> neg cosine sim': valid_metrics["negative_cosine_similarity"].avg,
                    })

        # clean up gpu cache
        del inputs, query_mask, document_mask, query_h, context_h
        gc.collect()
        torch.cuda.empty_cache()

        return valid_losses.avg
