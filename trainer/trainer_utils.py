import configuration as configuration
import torch
import torch.nn as nn
import numpy as np
import transformers
import more_itertools
from torch.utils.data import Sampler, Dataset, DataLoader
from torch import Tensor
from typing import List, Tuple, Dict, Type, Union
from utils.helper import seed_worker


def get_model_layers(model: nn.Module) -> List[nn.Module]:
    """ Get specific layers of each architecture of model

    Args:
        model: model instance from PLM for supervised fine-tune

    """
    layers = []
    attrs = [attr for attr in model.__dict__ if not callable(getattr(model, attr))]
    if 'encoder' in attrs:  # Encoder Model: BERT, RoBERTa, ALBERT, ELECTRA, DeBERTa, BigBird, Longformer, Reformer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)

    elif 'h' in attrs:  # Decoder Model: GPT2
        layers = [model.model.wte] + [model.model.wpe] + list(model.model.h)

    elif 'layers' in attrs:  # Decoder Model: LLAMA
        layers = [model.model.embed_tokens] + list(model.model.layers)

    layers.reverse()
    return layers


def get_optimizer_grouped_parameters(
    model: nn.Module,
    layerwise_lr: float,
    layerwise_weight_decay: float,
    layerwise_lr_decay: float
) -> List[Dict[str, float]]:
    """ Grouped Version Layer-wise learning rate decay

    This function implemented for fine-tuning pre-trained model to task specific model or domain specific model
    So not used in pre-training phase, in pre-training phase, you can use only pure optimizer without layer-wise lr decay

    1) select specific layers of each architecture of model: (get_model_layers())
      - encoder type (bert, roberta, albert, electra, deberta, bigbird, longformer, reformer)
        - embedding layer: model.embeddings
        - encoder: model.encoder

      - decoder type: gpt2, t5, bart, pegasus, prophetnet, reformer, blenderbot
        - word token embedding: model.wte
        - position embedding: model.wpe
        - decoder: model.h

      - llama:
        - embedding layer: model.embed_tokens
        - decoder: model.layers

    2) initialize lr for task specific layer
    3) initialize lr for every layer

    Args:
        model: model instance from customizing model
        layerwise_lr: learning rate for task specific layer
        layerwise_weight_decay: weight decay for task specific layer
        layerwise_lr_decay: learning rate decay for every layer

    """
    no_decay = ["bias", "LayerNorm.bias"]  # LayerNorm.weight
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "weight_decay": 0.0,
            "lr": layerwise_lr
        },
    ]

    layers = get_model_layers(model)
    lr = layerwise_lr
    for layer in layers:
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": layerwise_weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        lr *= layerwise_lr_decay
    return optimizer_grouped_parameters


def collate(inputs: Dict) -> Dict:
    """ Descending sort inputs by length of sequence

    Args:
        inputs: inputs from torch.utils.data.DataLoader, which is from transformers.AutoTokenizer
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def get_dataloader(
    cfg: configuration.CFG,
    dataset: Dataset,
    generator: torch.Generator,
    shuffle: bool = True,
    collate_fn=None,
    sampler=None,
    drop_last: bool = True
) -> DataLoader:
    """ function for initiaflizing torch.utils.data.DataLoader Module
    All Args are from torch.nn.utils.data.DataLoader except cfg
    """
    dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=drop_last
    )
    return dataloader


def get_scheduler(
    cfg: configuration.CFG,
    optimizer: torch.optim.Optimizer,
    len_train: int
) -> torch.optim.lr_scheduler:
    """ Select Scheduler Function (cosine_annealing, cosine, linear)

    Args:
        cfg: for getting scheduler options from configuration.py
        optimizer: torch.optim.Optimizer
        len_train: length of training dataset for calculating total steps
    """
    lr_scheduler = None
    scheduler_dict = {
        'cosine_annealing': 'get_cosine_with_hard_restarts_schedule_with_warmup',
        'cosine': 'get_cosine_schedule_with_warmup',
        'linear': 'get_linear_schedule_with_warmup',
        'constant': 'get_constant_schedule',
        'constant_with_warmup': 'get_constant_schedule_with_warmup',
    }
    if cfg.scheduler == 'cosine_annealing' or cfg.scheduler == 'cosine':
        lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
            optimizer,
            num_warmup_steps=int(len_train / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
            num_training_steps=int(len_train / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps),
            num_cycles=cfg.num_cycles
        )
    elif cfg.scheduler == 'linear':
        lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
            optimizer,
            num_warmup_steps=int(len_train / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
            num_training_steps=int(len_train / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps)
        )
    elif cfg.scheduler == 'constant':
        lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
            optimizer
        )
    elif cfg.scheduler == 'constant_with_warmup':
        lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
            optimizer,
            num_warmup_steps=int(len_train / cfg.batch_size * cfg.epochs / cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio
        )
    return lr_scheduler


def get_swa_scheduler(
    cfg: configuration.CFG,
    optimizer: torch.optim.Optimizer
) -> torch.optim.swa_utils.SWALR:
    """ SWA Scheduler
    Init scheduler for Stochastic Weight Averaging
    All Args are from torch.optim.swa_utils.SWALR except cfg

    Args:
        cfg: for getting SWA scheduler options from configuration.py
        optimizer: torch.optim.swa_utils.SWALR
    """
    swa_scheduler = getattr(torch.optim.swa_utils, 'SWALR')(
        optimizer,
        swa_lr=cfg.swa_lr,
        anneal_epochs=cfg.anneal_epochs,
        anneal_strategy=cfg.anneal_strategy
    )
    return swa_scheduler


def get_name(cfg: configuration.CFG) -> str:
    """ get name of model for recording experiment result """
    try:
        name = cfg.model.replace('/', '-')
    except ValueError:
        name = cfg.model
    return name


class SmartBatchingSampler(Sampler):
    """
    SmartBatching Sampler with naive pytorch torch.utils.data.Sampler implementation, which is iterable-style dataset
    not map-style dataset module.
    This class chunk whole of batch instances by length for making mini-batch by its length

    Args:
        data_instance: whole of batch instances, not mini-batch-level
        batch_size: amount of mini-batch from configuration.py or cfg.json

    Reference:
        https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies
    """
    def __init__(self, data_instance: Tensor, batch_size: int) -> None:
        super(SmartBatchingSampler, self).__init__(data_instance)
        self.len = len(data_instance)
        sample_lengths = [len(seq) for seq in data_instance]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self) -> int:
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)

        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self) -> int:
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds


class SmartBatchingCollate:
    """
    SmartBatchingCollate will add padding upto highest sequence length, make attention masks,
    targets for each sample in batch.

    Args:
        labels: target value of training dataset
        max_length: value from configuration.py or cfg.json which is initialized by user
        pad_token_id: int value from pretrained tokenizer, AutoTokenizer.tokenizer.pad_token_ids

    Reference:
        https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies
    """
    def __init__(self, labels: Tensor, max_length: int, pad_token_id: int) -> None:
        self._labels = labels
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch: Tuple) -> Tuple[Dict, Tensor]:
        if self._labels is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        output = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if self._labels is not None:
            labels = torch.as_tensor(targets)
        else:
            output = input_ids, attention_mask
        return output, labels

    @staticmethod
    def pad_sequence(sequence_batch: List, max_sequence_length: int, pad_token_id: int) -> Tuple[Tensor, Tensor]:
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        padded_sequences = torch.as_tensor(padded_sequences)
        attention_masks = torch.as_tensor(attention_masks)
        return padded_sequences, attention_masks


class MiniBatchCollate(object):
    """ Collate class for torch.utils.data.DataLoader
    This class object to use variable data such as NLP text sequence
    If you use static padding with AutoTokenizer, you don't need this class object
    But if you use dynamic padding with AutoTokenizer, you must use this class object & call

    Args:
        batch: data instance from torch.utils.data.DataSet
    """
    def __init__(self, batch: torch.utils.data.DataLoader) -> None:
        self.batch = batch

    def __call__(self) -> Tuple[Dict[Tensor, Tensor], Tensor, Tensor]:
        inputs, labels, position_list = self.batch
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-1
        )
        position_list = torch.nn.utils.rnn.pad_sequence(
            position_list,
            batch_first=True,
            padding_value=-1
        )
        return inputs, labels, position_list


class AWP:
    """ Adversarial Weight Perturbation for OneToOne Trainer

    Args:
        model: model instance from customizing model
        criterion: losses function for training, you must pass instance which is inheritance of torch.nn.Module
        optimizer: optimizer for training, you must pass instance which is inheritance of torch.optim or transformers.optimization
        apex: if you use apex, you must pass True
        adv_param: parameter name for adversarial weight perturbation, default is 'weight'
        adv_lr: learning rate for adversarial weight perturbation, default is 1.0
        adv_eps: epsilon for adversarial weight perturbation, default is 0.01

    References:
        https://www.kaggle.com/code/skraiii/pppm-tokenclassificationmodel-train-8th-place
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: Type[nn.Module],
        optimizer: torch.optim.Optimizer,
        apex: bool,
        adv_param: str = "weight",
        adv_lr: float = 1.0,
        adv_eps: float = 0.01
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: Dict, padding_mask: Tensor, label: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step()
            y_preds = self.model(inputs, padding_mask)
            adv_loss = self.criterion(
                y_preds.view(-1, 1), label.view(-1, 1))
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AverageMeter(object):
    """ Computes and stores the average and current value

    Reference:
        https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """ Monitor a metrics and stop training when it stops improving.

    Args:
        mode: 'min' for losses base val_score for losses, 'max' for metrics base val_score
        patience: number of checks with no improvement, default = 3
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement. default = 0.0
        detect_anomaly: When set ``True``, stops training when the monitor becomes NaN or infinite, etc
            default = True
    """
    def __init__(self, mode: str, patience: int = 3, min_delta: float = 0.0, detect_anomaly: bool = True) -> None:
        self.mode = mode
        self.early_stop = False
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.detect_anomaly = detect_anomaly
        self.val_score = -np.inf
        if self.mode == 'min':
            self.val_score = np.inf

    def detecting_anomaly(self) -> None:
        """ Detecting Trainer's Error and Stop train loop """
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        return

    def __call__(self, score: float) -> None:
        """ When call by Trainer Loop, Check Trainer need to early stopping """
        if self.mode == 'min':
            if self.val_score >= score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.mode == 'max':
            if score >= self.val_score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print('Early STOP')
