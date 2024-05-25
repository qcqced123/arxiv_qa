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
from dataset_class.preprocessing import load_all_types_dataset
from trainer.trainer_utils import load_pretrained_weights, get_optimizer_grouped_parameters, get_scheduler
from trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler


class PreTrainTuner:
    """ Trainer class for Pre-Train Pipeline, such as MLM
    So, if you want set options, go to cfg.json file or configuration.py

    Args:
        cfg: configuration module, confriguration.py
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
        """ Function for init backbone's configuration & train utils setting,
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
        """ function for train loop with validation for each batch*N Steps
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

            # logging train loss, gradient norm, lr to wandb
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
        """ function for train loop
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
        """ function for train loop with validation for each batch*N Steps """
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

            # logging train loss, gradient norm, lr to wandb
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


class TextGenerationTuner(PreTrainTuner):
    """ Fine-tune class for Text Generation, Summarization
    """
    def __init__(self, cfg: CFG, generator: torch.Generator, es: Elasticsearch = None) -> None:
        super(TextGenerationTuner, self).__init__(cfg=cfg, generator=generator)
        self.es = es

    def make_batch(self):
        """ for rag-token generation, you need to use different dataloader """
        pass

    def model_setting(self, len_train: int = None):
        """ Function for init backbone's configuration & train utils setting,
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)

        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            load_pretrained_weights(model, self.cfg)

        model.to(self.cfg.device)
        return model

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
    ):
        pass

    def valid_fn(
        self,
        loader_valid,
        model: nn.Module,
        val_criterion: nn.Module,
        val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ function for validation loop
        """

    @staticmethod
    def log_probs_from_logit(logits, labels):
        """ function for returning single-token sequence """
        logp = F.log_softmax(logits, dim=-1)
        logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logp_label

    def sequence_probs(self, model, labels, input_len: int = 0):
        """
        Args:
            model: nn.Module, PLM for generating text
            labels: torch.Tensor, convert input string into pytorch tensor
            input_len: int, length of each batch sequence
        """
        with torch.no_grad():
            output = model(labels)
            log_probs = self.log_probs_from_logit(output.logits[:, :-1, :], labels[:, 1:])  # rotate
            seq_log_prob = torch.sum(log_probs[:, input_len:])  # except last index token for calculating prob

        return seq_log_prob.cpu().numpy()

    def inference(
        self,
        model: nn.Module,
        query: str,
        context: str,
        max_length: int,
        strategy: str = None,
        penalty_alpha: float = None,
        num_beams: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
        length_penalty: float = None,
        no_repeat_ngram_size: int = None,
        do_sample: bool = None,
        use_cache: bool = True,
    ) -> List[Dict]:
        prompt = f"[query]\n{query}\n\n[context]\n{context}\n\n"
        input_ids = self.tokenizer(prompt, max_length=max_length, return_tensors='pt')['input_ids'].to(self.cfg.device)
        output = model.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            penalty_alpha=penalty_alpha,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        # logp = self.sequence_probs(
        #     model=model,
        #     labels=output,
        #     input_len=len(input_ids[0])
        # )
        # print(f"Decoding Probability: {logp}")
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class SequenceClassificationTuner:
    """ Trainer class for baseline fine-tune pipeline, such as text classification, sequence labeling, etc.

    Text Generation, Question Answering, Text Similarity and so on are not supported on this class
    They may be added another module, inherited from this class
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.fold_value = 3  # you can change any other value: range 0 to 9
        self.generator = generator
        self.model_name = self.cfg.model_name
        self.tokenizer = self.cfg.tokenizer
        self.metric_list = self.cfg.metrics

    def make_batch(self) -> Tuple[DataLoader, DataLoader, int]:
        base_path = './dataset_class/data_folder/'
        df = load_all_types_dataset(base_path + self.cfg.domain + 'train.csv')

        train = df[df['fold'] != self.fold_value]
        valid = df[df['fold'] == self.fold_value]

        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid)

        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            collate_fn=self.cfg.collator,
            sampler=self.cfg.sampler,
            generator=self.generator,
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            collate_fn=self.cfg.collator,
            sampler=self.cfg.sampler,
            generator=self.generator,
            shuffle=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(train)

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & train utils setting,
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

        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.weight_decay,
            self.cfg.layerwise_lr_decay
        )

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.adam_epsilon,
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
        """ train method with step-level validation
        """
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
                loss = criterion(logit.view(-1, self.cfg.num_labels), labels.view(-1))

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

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Total Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # step-level validation
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                d_valid_loss, s_valid_loss, c_valid_loss = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                valid_loss = d_valid_loss + s_valid_loss + c_valid_loss
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
    ) -> Tuple[float, float, float]:
        """ validation method for sentence(sequence) classification task
        """
        valid_losses = AverageMeter()
        y_pred, y_true = np.array([]), np.array([])
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)

                batch_size = labels.size(0)

                logit = model(inputs)
                flat_logit, flat_label = logit.view(-1, self.cfg.num_labels), labels.view(-1)
                loss = val_criterion(flat_logit, flat_label)

                valid_losses.update(loss.item(), batch_size)
                wandb.log({'<Val Step> Valid Loss': valid_losses.avg})

                flat_logit, flat_label = flat_logit.detach().cpu().numpy(), flat_label.detach().cpu().numpy()
                y_pred, y_true = np.append(y_pred, np.argmax(flat_logit, axis=-1)), np.append(y_true, flat_label)
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(
                        flat_label,
                        flat_logit,
                        self.cfg
                    )
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Valid {self.metric_list[i]}': valid_metrics[self.metric_list[i]].avg,
                    })

                # plotting confusion matrix to wandb
                wandb.log({'<Val Step> Valid confusion matrix:': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[f"Rating {i + 1}" for i in range(self.cfg.num_labels)]
                )})
        return valid_losses.avg
