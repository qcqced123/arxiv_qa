import torch.nn as nn
from configuration import CFG
from typing import List, Dict
from elasticsearch import Elasticsearch
from db.run_db import search_candidates
from transformers import AutoTokenizer


def query_encoder(
    cfg: CFG,
    retriever: nn.Module,
    tokenizer: AutoTokenizer,
    query: str = "",
    top_k: int = 5,
    es: Elasticsearch = None,
) -> str:
    """ function for encoding input query and searching the nearest top-k documents from document embedding db

    Args:
        cfg (CFG): configuration module
        retriever (nn.Module): embedding mapper of input texts
        tokenizer (AutoTokenizer): module of tokenizing the inputs
        es (Elasticsearch): elastic search engine
        query (str): input query for searching
        top_k (int): number of top k candidates
    """
    result = search_candidates(
        cfg=cfg,
        encoder=retriever,
        tokenizer=tokenizer,
        query=query,
        es=es,
        top_k=top_k
    )["inputs"]
    return "\n".join(result)
