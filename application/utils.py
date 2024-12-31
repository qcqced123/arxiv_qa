"""utils module for running application
"""
import platform


from configuration import CFG
from db.helper import get_tokenizer
from typing import List, Dict, Tuple
from application.model import QueryList
from elasticsearch import Elasticsearch
from trainer.train_loop import inference_loop
from db.run_db import get_encoder, run_engine
from db.run_db import get_db_url, get_db_auth, get_db_cert
from inference.vllm_inference import initialize_llm, get_sampling_params


def initialize_es() -> Elasticsearch:
    """ initialize the Elastic Search module for finding candidates document to answering the questions from users

    Return:
        Elastic Search Engine module
    """
    os_type = platform.system()
    return run_engine(
        url=get_db_url(),
        auth=get_db_auth(os_type),
        cert=get_db_cert(os_type)
    )


def initialize_retriever(cfg: CFG) -> Dict:
    """ initialize the module for retriever like as tokenizer, model object and load them to target accelerator
    Args:

    Return:
        Dict
    """
    return {
        "retriever_tokenizer": get_tokenizer(cfg.retriever_name),
        "retriever": get_encoder(cfg.retriever_name).to(cfg.device)
    }


def initialize_generator(cfg: CFG) -> Dict:
    """ initialize the module for generator like as tokenizer, model object and load them to target accelerator
    Args:

    Return:
        Dict
    """
    return {
        "generator_tokenizer": get_tokenizer(cfg.generator_name),
        "generator": initialize_llm(cfg=cfg),
        "sampling_params": get_sampling_params(cfg)
    }


def make_queries(queries: QueryList) -> List[str]:
    """ post user's queries to question encoder module for answering about questions
    Args:
        queries (QueryList): module for QueryList, containing the user's query list

    Returns:
        python list object of user's queries for making the input queries of query encoder
    """
    return [query.question for query in queries.queries]


def make_templates(queries: List[str], answers: List[str]) -> List[Dict]:
    """ function for making answer templates and returning iterable object to main interface
    Args:
        queries (List[str]):
        answers (List[str]):

    Return:
        List of containing dictionary, query and answer key-value pair
    """
    return [{"question": q, "answer": a} for q, a in zip(queries, answers)]


def call_inference_api(
    cfg: CFG,
    retriever_dict: Dict,
    generator_dict: Dict,
    es: Elasticsearch,
    queries: List[str]
):
    """ call the inference function to answering the input queries from web users,
    all arguments of this caller are exactly same as callee (inference_loop())

    callee will return the list of answers for each queries from users

    """
    return inference_loop(
        cfg=cfg,
        retriever_dict=retriever_dict,
        generator_dict=generator_dict,
        es=es,
        queries=queries
    )
