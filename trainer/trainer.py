import gc
import wandb
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F

import trainer.loss as loss
import trainer.metric as metric
import dataset_class.dataclass as dataset_class

import vllm
from vllm import LLM, SamplingParams

from transformers import AutoConfig, AutoTokenizer

from tqdm.auto import tqdm
from typing import Union, List, Callable, Dict

from transformers import AutoConfig
from elasticsearch import Elasticsearch

from configuration import CFG
from model import model as task
from model import clm

from db.run_db import get_encoder
from dataset_class.preprocessing import dataset_split, load_all_types_dataset

from trainer.collator_fn import BatchCollatorFunc
from trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler, get_name
from trainer.trainer_utils import load_pretrained_weights, get_optimizer_grouped_parameters, get_scheduler


class TextGenerationTuner:
    """ inference pipeline module for text generation, generative text summarization task

    Args:
        cfg: configuration module, configuration.py
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

        return retriever, model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler

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

    def trt_llm_inference(
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
        pass

    def vllm_inference(
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
        """ class method for inference pipeline for vllm, which can apply the paged attention, kv-cache,
        in-flight batching ...

        this method is designed for the faster generative task, such as summarization, text generation, ...
        than native pytorch & huggingface inference pipeline
        """
        llm = LLM(
            model=self.cfg.model_name,
            max_model_len=6144,
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        sampling_config = SamplingParams(
            best_of=1,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=self.cfg.seed,
            early_stopping=True,
            detokenize=True,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        output = llm.generate(prompt, sampling_config)
        return output

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

        this method is designed with native pytorch & huggingface library,
        if you want to use other faster platform such as tensorrt_llm, vllm, you must change the config value,
        named "inference_pipeline"

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
        return loader_train, loader_valid, len(self.df)

    def model_setting(self, len_train: int) -> Union:
        # initialize and get the model module
        # transfer model in CPU to GPU
        model = getattr(task, self.cfg.task)(self.cfg)
        model.to(self.cfg.device)

        # initialize and get the loss, metric module
        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
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
                    '<Val Step> Valid Loss': valid_losses.avg
                })

                # we calculate the top-1 accuracy for each batch instance
                # for calculating the top-1 accuracy
                queries, contexts = query_h.detach().cpu().numpy(), context_h.detach().cpu().numpy()
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

        # clean up gpu cache
        del inputs, query_mask, document_mask, query_h, context_h
        gc.collect()
        torch.cuda.empty_cache()

        return valid_losses.avg
