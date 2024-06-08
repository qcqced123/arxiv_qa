import gc
import wandb
import torch
from torch.optim.swa_utils import update_bn
import numpy as np
import trainer.trainer as trainer

from tqdm.auto import tqdm
from typing import List, Dict
from elasticsearch import Elasticsearch

from configuration import CFG
from utils.helper import class2dict
from trainer.trainer import TextGenerationTuner
from trainer.trainer_utils import get_name, EarlyStopping
from db.run_db import get_encoder
from query_encoder.query_encoder import query_encoder

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: CFG, pipeline_type: str, model_config: str, es: Elasticsearch) -> None:
    """ Base Trainer Loop Function
    1) Initialize Trainer Object
    2) Make Early Stopping Object
    3) Initialize Metric Checker
    4) Initialize Train, Validation Input Object
    5) Check if this insert loop need to finish, by Early Stopping Object
    """

    sub_name = f""
    group_name = f"{pipeline_type}/"

    wandb.init(
        project=cfg.name,
        name=sub_name,
        config=class2dict(cfg),
        group=group_name,
        job_type=cfg.pipeline_type,
        entity="qcqced"
    )

    early_stopping = EarlyStopping(mode=cfg.stop_mode, patience=cfg.patience)
    early_stopping.detecting_anomaly()  # call detecting anomaly in pytorch

    metric_checker = []
    for _ in range(3):
        if cfg.stop_mode == 'min':
            metric_checker.append(np.inf)

        elif cfg.stop_mode == 'max':
            metric_checker.append(-np.inf)

    epoch_val_score_max, val_score_max, val_score_max_2 = metric_checker
    train_input = getattr(trainer, cfg.trainer)(cfg, g, es)  # init object

    loader_train, loader_valid, len_train = train_input.make_batch()
    retriever, model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler = train_input.model_setting(
        len_train
    )
    train_val_method = train_input.train_val_fn
    for epoch in tqdm(range(cfg.epochs)):
        print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
        train_loss, val_score_max = train_val_method(
            loader_train,
            retriever,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            loader_valid,
            val_criterion,
            val_metric_list,
            val_score_max,
            val_score_max_2,
            epoch,
            awp,
        )
        wandb.log({
            '<epoch> Train Loss': train_loss,
            '<epoch> Valid Loss': val_score_max,
        })
        print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
        print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(val_score_max, 4)}')

        if epoch_val_score_max >= val_score_max:
            print(f'Best Epoch Score: {val_score_max}')
            epoch_val_score_max = val_score_max
            wandb.log({
                '<epoch> Valid Loss': val_score_max,
            })
            print(f'Valid Best Loss: {np.round(val_score_max, 4)}')

        # Check if Trainer need to Early Stop
        early_stopping(val_score_max)
        if early_stopping.early_stop:
            break

        del train_loss
        gc.collect(), torch.cuda.empty_cache()

        if cfg.swa and not early_stopping.early_stop:
            update_bn(loader_train, swa_model)
            swa_loss = train_input.swa_fn(loader_valid, swa_model, val_criterion)
            print(f'[{epoch + 1}/{cfg.epochs}] SWA Val Loss: {np.round(swa_loss, 4)}')
            wandb.log({'<epoch> SWA Valid Loss': swa_loss})
            if val_score_max >= swa_loss:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {swa_loss:.4f}) Save Parameter')
                print(f'Best Score: {swa_loss}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}_CV_{swa_loss}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                wandb.log({'<epoch> Valid Loss': swa_loss})

    wandb.finish()


def inference_loop(cfg: CFG, pipeline_type: str, model_config: str, es: Elasticsearch) -> List[str]:
    queries = [
        "What is the self-attention mechanism in transformer?",
        "What is the Retrieval Augmented Generation (RAG) model?",
    ]  # ASAP, this line will be changed to user's input q
    retriever = get_encoder(cfg.retriever)
    tuner = TextGenerationTuner(
        cfg=cfg,
        generator=g,
        is_train=False,
        es=es
    )
    _, generator, *_ = tuner.model_setting()

    answers = []  # retrieve the top-k documents from the elastic search engine
    for query in queries:
        prompt = query_encoder(
            es=es,
            retriever=retriever,
            query=query,
            top_k=10
        )

        answer = tuner.inference(
            prompt=prompt,
            model=generator,
            max_new_tokens=cfg.max_new_tokens,
            max_length=cfg.max_len,
            return_full_text=cfg.return_full_text,
            strategy=cfg.strategy,
            penalty_alpha=cfg.penalty_alpha,
            num_beams=cfg.num_beams,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            length_penalty=cfg.length_penalty,
            do_sample=cfg.do_sample,
            use_cache=cfg.use_cache
        )
        answers.append(answer)

    return answers

