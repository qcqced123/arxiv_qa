import wandb
import numpy as np
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F

import trainer.loss as loss
import trainer.metric as metric
import dataset_class.dataclass as dataset_class

from torch import Tensor
from numpy import ndarray
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from typing import Tuple, Any, Union, List, Callable, Dict

from elasticsearch import Elasticsearch

from configuration import CFG
from model import model as task
from model import mlm, clm, sbo

from db.run_db import get_encoder
from dataset_class.preprocessing import dataset_split, load_all_types_dataset

from trainer.loss import MultipleNegativeRankingLoss
from trainer.collator_fn import CollatorFunc
from trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler
from trainer.trainer_utils import load_pretrained_weights, get_optimizer_grouped_parameters, get_scheduler


class PreTrainTuner:
    """ Trainer class for Pre-Train Pipeline, such as MLM
    So, if you want set options, go to cfg.json file or configuration.py

    Args:
        cfg: configuration module, configurations.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.module_name
        self.tokenizer = self.cfg.tokenizer
        self.generator = generator
        self.metric_list = self.cfg.metrics

    def make_batch(self) -> Tuple[DataLoader, DataLoader, int]:
        """ Function for making batch instance
        """
        train = load_all_types_dataset(f'./dataset_class/data_folder/{self.cfg.datafolder}/train_text.pkl')
        valid = load_all_types_dataset(f'./dataset_class/data_folder/{self.cfg.datafolder}/valid_text.pkl')

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(valid, is_valid=True)

        # 2) selecting custom masking method for each task
        collate_fn = None
        if self.cfg.task in ['MaskedLanguageModel', 'DistillationKnowledge']:
            collate_fn = getattr(mlm, 'MLMCollator')(self.cfg)
        elif self.cfg.task == 'CasualLanguageModel':
            collate_fn = getattr(clm, 'CLMCollator')(self.cfg)
        elif self.cfg.task == 'SpanBoundaryObjective':
            collate_fn = getattr(sbo, 'SpanCollator')(
                self.cfg,
                self.cfg.masking_budget,
                self.cfg.span_probability,
                self.cfg.max_span_length,
            )
        elif self.cfg.task == 'ReplacedTokenDetection':
            if self.cfg.rtd_masking == 'MaskedLanguageModel':
                collate_fn = getattr(mlm, 'MLMCollator')(self.cfg)
            elif self.cfg.rtd_masking == 'SpanBoundaryObjective':
                collate_fn = getattr(sbo, 'SpanCollator')(
                    self.cfg,
                    self.cfg.masking_budget,
                    self.cfg.span_probability,
                    self.cfg.max_span_length,
                )

        # 3) Initializing torch.utils.data.DataLoader Module
        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            collate_fn=collate_fn,
            generator=self.generator
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            collate_fn=collate_fn,
            generator=self.generator,
            shuffle=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(train['input_ids'])

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & insert utils setting,
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)

        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict),
                strict=False
            )
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
        )
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        # init SWA Module
        swa_model, swa_scheduler = None, None
        if self.cfg.swa:
            swa_model = AveragedModel(model)
            swa_scheduler = get_swa_scheduler(self.cfg, optimizer)

        # init AWP Module
        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )
        return model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

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
            val_score_max_2: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        """ function for insert loop with validation for each batch*N Steps
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()

        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad(set_to_none=True)
            inputs = batch['input_ids'].to(self.cfg.device, non_blocking=True)
            labels = batch['labels'].to(self.cfg.device, non_blocking=True)
            padding_mask = batch['padding_mask'].to(self.cfg.device, non_blocking=True)
            batch_size = inputs.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                logit = model(inputs, padding_mask)
                loss = criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.item(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:  # adv training option
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            # update parameters
            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )

                # Stochastic Weight Averaging option
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # logging insert loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # validate for each size of batch*N Steps
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss = self.valid_fn(loader_valid, model, val_criterion, val_metric_list)
                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}{self.cfg.mlm_masking}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None, swa_model=None, swa_start=None, swa_scheduler=None) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for insert loop
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()

        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad(set_to_none=True)
            inputs = batch['input_ids'].to(self.cfg.device, non_blocking=True)
            labels = batch['labels'].to(self.cfg.device, non_blocking=True)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device, non_blocking=True)  # padding mask to GPU
            batch_size = inputs.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                logit = model(inputs, padding_mask)
                loss = criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        grad_norm = grad_norm.detach().cpu().numpy()
        return losses.avg, grad_norm, scheduler.get_lr()[0]

    def valid_fn(
            self,
            loader_valid,
            model: nn.Module,
            val_criterion: nn.Module,
            val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ function for validation loop
        """
        valid_losses = AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device, non_blocking=True)
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)
                padding_mask = batch['padding_mask'].to(self.cfg.device, non_blocking=True)
                batch_size = inputs.size(0)

                logit = model(inputs, padding_mask)
                flat_logit, flat_labels = logit.view(-1, self.cfg.vocab_size), labels.view(-1)

                loss = val_criterion(flat_logit, flat_labels)
                valid_losses.update(loss.item(), batch_size)

                wandb.log({
                    '<Val Step> Valid Loss': valid_losses.avg
                })

                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(
                        flat_labels.detach().cpu().numpy(),
                        flat_logit.detach().cpu().numpy()
                    )
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Valid {self.metric_list[i]}': valid_metrics[self.metric_list[i]].avg
                    })
        return valid_losses.avg

    def swa_fn(
            self,
            loader_valid,
            swa_model,
            val_criterion,
            val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ Stochastic Weight Averaging, it consumes more GPU VRAM & training times
        """
        valid_losses = AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}

        swa_model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device, non_blocking=True)
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)  # Two target values to GPU
                padding_mask = batch['padding_mask'].to(self.cfg.device, non_blocking=True)  # padding mask to GPU
                batch_size = inputs.size(0)

                logit = swa_model(inputs, padding_mask)
                flat_logit, flat_labels = logit.view(-1, self.cfg.vocab_size), labels.view(-1)

                loss = val_criterion(flat_logit, flat_labels)
                valid_losses.update(loss.item(), batch_size)
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(flat_labels.detach().cpu().numpy(), flat_logit.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores


class CLMTuner(PreTrainTuner):
    """ Trainer class for Casual Language Model Task, such as transformer decoder, GPT2, GPTNeo, T5 ...
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        super(CLMTuner, self).__init__(cfg, generator)

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
        val_score_max_2: float,
        epoch: int,
        awp: nn.Module = None,
        swa_model: nn.Module = None,
        swa_start: int = None,
        swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        """ function for insert loop with validation for each batch*N Steps """
        losses = AverageMeter()
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)

        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.cfg.device, non_blocking=True)
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                logit = model(inputs)
                loss = criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))  # cross entropy loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.item(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                adv_loss = awp.attack_backward(inputs['input_ids'], inputs['attention_mask'], labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            grad_norm = None
            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                ).item()
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # logging insert loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            wandb.log({
                '<Per Step> Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm if grad_norm is not None else 0,
                '<Per Step> lr': lr,
            })

            # validate for each size of batch*N Steps
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss = self.valid_fn(loader_valid, model, val_criterion, val_metric_list)
                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}{self.cfg.mlm_masking}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def valid_fn(
            self,
            loader_valid,
            model: nn.Module,
            val_criterion: nn.Module,
            val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ function for validation loop
        """
        valid_losses = AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)
                batch_size = labels.size(0)

                with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                    logit = model(inputs)
                    flat_logit, flat_labels = logit.view(-1, self.cfg.vocab_size), labels.view(-1)
                    loss = val_criterion(flat_logit, flat_labels)

                valid_losses.update(loss.item(), batch_size)

                wandb.log({
                    '<Val Step> Valid Loss': valid_losses.avg
                })
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(valid_losses.avg) if not i else metric_fn(flat_labels.detach().cpu().numpy(), flat_logit.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Valid {self.metric_list[i]}': valid_metrics[self.metric_list[i]].avg
                    })
        return valid_losses.avg


class TextGenerationTuner:
    """ Fine-tune class for Text Generation, Summarization

    Args:
        cfg: configuration module, confriguration.py
        generator: torch.Generator, for init pytorch random seed
        is_train: bool, for setting train or validation mode
        es: Elasticsearch, for searching the top-k nearest document in doc embedding db
    """
    def __init__(self, cfg: CFG, generator: torch.Generator, is_train: bool = True, es: Elasticsearch = None) -> None:
        self.es = es
        self.cfg = cfg
        self.generator = generator
        self.is_train = is_train
        self.metric_list = self.cfg.metrics
        self.tokenizer = self.cfg.tokenizer
        self.model_name = self.cfg.module_name

    def make_batch(self):
        """ search the top-k nearest document in doc embedding db for making the input prompt of generator
        by using retriever and query from the generative Question and Answering DataFrame
        """
        df = load_all_types_dataset("./dataset_class/arxiv_qa/train_paper_meta_db.csv") if self.is_train else None
        train, valid = dataset_split(self.cfg, df)

        train_dataset = getattr(dataset_class, self.cfg.dataset)(train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(valid, is_valid=True)

        collate_fn = getattr(clm, 'CLMCollator')(self.cfg)
        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            collate_fn=collate_fn,
            generator=self.generator
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            collate_fn=collate_fn,
            generator=self.generator,
            shuffle=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(train['input_ids'])

    def model_setting(self, len_train: int = None):
        """ Function for init backbone's configuration & insert utils setting,
        The design is inspired by the Builder Pattern
        """
        retriever = None
        if self.cfg.pipeline_type == 'inference':
            retriever = get_encoder(self.cfg.retriever)
            retriever.to(self.cfg.device)

        model = getattr(task, self.cfg.task)(self.cfg)

        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            load_pretrained_weights(model, self.cfg)

        model.to(self.cfg.device)

        criterion, val_criterion, val_metric_list = None, None, None
        optimizer, lr_scheduler = None, None
        swa_model, swa_scheduler, awp = None, None, None

        if self.cfg.pipeline_type == 'insert':
            criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
            val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
            val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

            optimizer = getattr(transformers, self.cfg.optimizer)(
                params=retriever.parameters() + model.parameters(),
                lr=self.cfg.lr,
                betas=self.cfg.betas,
                eps=self.cfg.adam_epsilon,
                weight_decay=self.cfg.weight_decay,
                correct_bias=not self.cfg.use_bertadam
            )
            lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

            # init SWA Module
            if self.cfg.swa:
                swa_model = AveragedModel(model)
                swa_scheduler = get_swa_scheduler(self.cfg, optimizer)

            # init AWP Module

            if self.cfg.awp:
                awp = AWP(
                    model,
                    criterion,
                    optimizer,
                    self.cfg.awp,
                    adv_lr=self.cfg.awp_lr,
                    adv_eps=self.cfg.awp_eps
                )
        return retriever, model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

    def train_val_fn(self) -> None:
        pass

    def valid_fn(self) -> None:
        pass

    @staticmethod
    def log_probs_from_logit(logits, labels):
        """ function for returning single-token sequence """
        logp = F.log_softmax(logits, dim=-1)
        logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logp_label

    @torch.no_grad()
    def sequence_probs(self, model, labels, input_len: int = 0):
        """
        Args:
            model: nn.Module, PLM for generating text
            labels: torch.Tensor, convert input string into pytorch tensor
            input_len: int, length of each batch sequence
        """
        output = model(labels)
        log_probs = self.log_probs_from_logit(output.logits[:, :-1, :], labels[:, 1:])  # rotate
        seq_log_prob = torch.sum(log_probs[:, input_len:])  # except last index token for calculating prob
        return seq_log_prob.cpu().numpy()

    @torch.no_grad()
    def inference(
        self,
        model: nn.Module,
        max_new_tokens: int,
        max_length: int,
        query: str = None,
        context: str = None,
        prompt: str = None,
        strategy: str = None,
        penalty_alpha: float = None,
        num_beams: int = None,
        temperature: float = 1,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = None,
        length_penalty: float = None,
        no_repeat_ngram_size: int = None,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> List[Dict]:
        """ method for making the answer from the given prompt (context + query) or pre-defined prompt by caller

        generate method's arguments setting guide:
            1) num_beams: 1 for greedy search, > 1 for beam search

            2) temperature: softmax distribution, default is 1 meaning that original softmax distribution
                            (if you set < 1, it will be more greedy, if you set > 1, it will be more diverse)

            3) do_sample: flag for using sampling method, default is False
                          (if you want to use top-k or top-p sampling, set this flag to True)

            4) top_k: top-k sampling, default is 50, must do_sample=True
            5) top_p: top-p (nucleus) sampling, default is 0.9, must do_sample=True
        """
        prompt = prompt if prompt is not None else f"context:{context}\nquery:{query}"

        inputs = self.tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
        for k, v in inputs.items():
            inputs[k] = torch.as_tensor(v)

        output = model.model.generate(
            input_ids=inputs['input_ids'].to(self.cfg.device),
            max_new_tokens=max_new_tokens,
            # max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            penalty_alpha=penalty_alpha,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=use_cache,
        )
        # output has nested tensor, so we need to flatten it for decoding
        result = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return result


class MetricLearningTuner:
    """ trainer class for conducting metric learning objective, such as contrastive learning,
    multiple negative ranking loss, arcface ...

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
        self.model_name = self.cfg.model_name
        self.tokenizer = self.cfg.tokenizer
        self.metric_list = self.cfg.metrics
        self.df = load_all_types_dataset(
            f'./dataset_class/datafolder/arxiv_qa/total/test_q_doc.csv'
        )

    def make_batch(self):
        self.df.dropna(subset=['question'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        train, valid = dataset_split(self.cfg, self.df)

        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid)

        collate_fn = CollatorFunc()
        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            collate_fn=collate_fn,
            generator=self.generator
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            collate_fn=collate_fn,
            generator=self.generator,
            shuffle=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(train)

    def model_setting(self, len_train: int):
        model = getattr(task, self.cfg.task)(self.cfg, len_train)

        if self.cfg.resume:
            model.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict),
                strict=False
            )
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.weight_decay,
            self.cfg.layerwise_lr_decay
        )

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
        )

        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        # init SWA Module
        swa_model, swa_scheduler = None, None
        if self.cfg.swa:
            swa_model = AveragedModel(model)
            swa_scheduler = get_swa_scheduler(self.cfg, optimizer)

        # init AWP Module
        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )
        return model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

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
        val_score_max_2: float,
        epoch: int,
        awp: nn.Module = None,
        swa_model: nn.Module = None,
        swa_start: int = None,
        swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        model.train()
        losses = AverageMeter()  # for logging total batch losses (arcface + contrastive loss)
        mnrl = MultipleNegativeRankingLoss()  # initialize the multiple negative ranking loss instance
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)

        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            query_index = batch['query_index'].to(self.cfg.device, non_blocking=True)
            document_index = batch['document_index'].to(self.cfg.device, non_blocking=True)
            labels = batch['labels'].to(self.cfg.device, non_blocking=True)

            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                query_h, context_h = model(
                    inputs=inputs,
                    query_index=query_index,
                    context_index=document_index
                )

                # do not call view, logit of arcface
                mnrl_loss = mnrl(query_h, context_h)
                arc_loss = criterion(model.arcface_head(query_h, labels), labels) + criterion(model.arcface_head(context_h, labels), labels)
                loss = mnrl_loss + arc_loss

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
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Total Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # branch for conducting validation stage when the current step value is same as config's "val_check" value
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                if val_score_max >= valid_loss:
                    print(f'[Update] Total Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Total Best Score: {valid_loss}')
                    torch.save(
                        model.model.student.state_dict(),
                        f'{self.cfg.checkpoint_dir}DistilBERT_Student_{self.cfg.mlm_masking}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def valid_fn(
        self,
        loader_valid,
        model: nn.Module,
        val_criterion: nn.Module,
        val_metric_list: List[Callable]
    ) -> float:
        """ validation method for sentence(sequence) classification task
        """
        model.eval()
        valid_losses = AverageMeter()
        mnrl = MultipleNegativeRankingLoss()  # initialize the multiple negative ranking loss instance
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                query_index = batch['query_index'].to(self.cfg.device, non_blocking=True)
                document_index = batch['document_index'].to(self.cfg.device, non_blocking=True)
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)

                batch_size = labels.size(0)

                query_h, context_h = model(
                    inputs=inputs,
                    query_index=query_index,
                    context_index=document_index
                )
                # do not call view, logit of arcface
                mnrl_loss = mnrl(query_h, context_h)
                arc_loss = val_criterion(model.arcface_head(query_h, labels), labels) + val_criterion(model.arcface_head(context_h, labels), labels)

                loss = mnrl_loss + arc_loss

                valid_losses.update(loss.item(), batch_size)
                wandb.log({'<Val Step> Valid Loss': valid_losses.avg})

                # we calculate the top-1 accuracy for each batch instance
                # for calculating the top-1 accuracy
                queries, contexts = query_h.detach().numpy(), context_h.detach().numpy()
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(
                        query=queries,
                        document=contexts,
                        k=1
                    )
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Valid {self.metric_list[i]}': valid_metrics[self.metric_list[i]].avg,
                    })
        return valid_losses.avg
