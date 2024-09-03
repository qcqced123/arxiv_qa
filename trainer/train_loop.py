import gc
import wandb
import torch
import numpy as np
import pandas as pd
import trainer.trainer as trainer

from tqdm.auto import tqdm
from typing import List, Dict
from elasticsearch import Elasticsearch

from configuration import CFG
from utils.helper import class2dict
from query_encoder.query_encoder import query_encoder
from trainer.trainer_utils import get_name, EarlyStopping

from inference.helper import apply_normalizer
from inference.post_process import slice_full_questions
from inference.vllm_inference import build_generating_prompt, build_answering_prompt, do_inference

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: CFG, pipeline_type: str, model_config: str) -> None:
    """ baseline trainer loop function, in this project this function will be used to fine-tune by metric-learning method

    Args:

    """

    sub_name = f"{cfg.model_name}"
    group_name = f"[{pipeline_type}]/"

    wandb.init(
        project=cfg.name,
        name=sub_name,
        config=class2dict(cfg),  # write and save the current experiment config state to weight and bias
        group=group_name,
        job_type=cfg.pipeline_type,
        entity="qcqced",
        resume=None
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
    train_input = getattr(trainer, cfg.trainer)(cfg, g)  # init object

    loader_train, loader_valid, len_train = train_input.make_batch()
    model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler = train_input.model_setting(
        len_train
    )
    train_val_method = train_input.train_val_fn
    for epoch in tqdm(range(cfg.epochs)):
        print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
        train_loss, val_score_max = train_val_method(
            loader_train,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            loader_valid,
            val_criterion,
            val_metric_list,
            val_score_max,
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

        # check if Trainer need to Early Stop
        early_stopping(val_score_max)
        if early_stopping.early_stop:
            break

        del train_loss
        gc.collect(), torch.cuda.empty_cache()

    wandb.finish()


def generate_loop(
    cfg: CFG,
    flow: str = "init",
    df: pd.DataFrame = None,
    generator_dict: Dict = None,
) -> None:
    """ generate function for making the question to each input context, originally from Arxiv paper pdf
    this function use the vllm generative backend for more faster inferencing than pure huggingface generative API

    Args:
        cfg (CFG): configuration module for inferencing
        flow (str):
        df (pd.DataFrame):
        generator_dict (Dict):

    Workflow:

    """
    # reference the necessary module for generating questions
    generator = generator_dict["generator"]
    sampling_params = generator_dict["sampling_params"]
    generator_tokenizer = generator_dict["generator_tokenizer"]
    text_normalizer = generator_dict["text_normalizer"]

    # clean and normalize the chunked text document from arxiv paper pdf
    # apply generative prompt template to cleansing and normalizing output
    document_list = [apply_normalizer(text_normalizer, document) for document in tqdm(df["doc"].tolist())] if flow == "init" else df["doc"].tolist()
    prompts = build_generating_prompt(
        tokenizer=generator_tokenizer,
        text_list=document_list
    )

    # chunk the total dataset instance for In-flight batching inference
    size = len(prompts) // 200
    chunked = [prompts[i:i + size] for i in range(0, len(prompts), size)]

    questions = []
    for sub in tqdm(chunked):
        outputs = do_inference(
            llm=generator,
            inputs=sub,
            sampling_params=sampling_params
        )
        questions.extend([slice_full_questions(output.outputs[0].text) for output in outputs])

    df["question"] = questions
    df.to_csv("output_arxiv_test.csv", index=False)

    return


def inference_loop(
    cfg: CFG,
    retriever_dict: Dict,
    generator_dict: Dict,
    es: Elasticsearch
) -> List[str]:
    """ inference function for making the answer to each input queries with elastic search, vllm backend

    Args:
        cfg (CFG): configuration module for inferencing
        retriever_dict (Dict):
        generator_dict (Dict):
        es (Elasticsearch):

    workflow:
        1) get multiple-inputs from multiple-users
        2) get candidates of each inputs from from document DB
        3) make the input prompt, using the inputs queries and candidates
            - build the input prompt by using chatting template in LLM models
            - find the optimal input prompt shape
        4) generate the answer from question, using the vllm backend
    """
    # get multiple-queries from multiple-users
    queries = [
        "What is the self-attention mechanism in transformer?",
        "What is the Retrieval Augmented Generation (RAG) model?",
        "Why the self-attention mechanism use the Q, K, V matrix?",
        "How to train the transformer model?",
        "How to implement the block-sparse attention from BigBird paper?",
        "Tell me about the more detail of self-attention mechanism in transformer",
        "Why the transformer model add the two different matrix, word embedding and positional encoding?",
        "What is the main concept of Activation Aware Quantization?",
    ]

    # reference the retriever module
    retriever = retriever_dict["retriever"]
    retriever_tokenizer = retriever_dict["retriever_tokenizer"]

    # retrieve the top-k documents from the elastic search engine
    # concatenate the retriever's result (top-k candidates for each query from users)
    candidates = [
        query_encoder(cfg=cfg, retriever=retriever, tokenizer=retriever_tokenizer, query=query, top_k=5, es=es) for query in tqdm(queries)
    ]

    # make the input prompt for generating answer from the users queries
    # do inference for users queries
    generator = generator_dict["generator"]
    generator_tokenizer = generator_dict["generator_tokenizer"]
    sampling_params = generator_dict["sampling_params"]

    prompts = build_answering_prompt(
        tokenizer=generator_tokenizer,
        queries=queries,
        candidates=candidates
    )

    size = len(prompts) // 4  # number of users
    chunked = [prompts[i:i + size] for i in range(0, len(prompts), size)]

    questions = []
    for sub in tqdm(chunked):
        outputs = do_inference(
            llm=generator,
            inputs=sub,
            sampling_params=sampling_params
        )
        questions.extend([output.outputs[0].text for output in outputs])

    return questions

