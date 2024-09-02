import pandas as pd
import torch.nn as nn

from configuration import CFG
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch

from db.run_db import create_index, insert_doc_embedding


def document_encoder(cfg: CFG, retriever: nn.Module, tokenizer: AutoTokenizer, es: Elasticsearch, df: pd.DataFrame) -> None:
    """ function for creating, inserting doc embedding into elastic search engine """
    try:
        create_index(cfg.model_name, es)

    except Exception as e:
        print(f"Error: {e}")

    insert_doc_embedding(
        cfg=cfg,
        encoder=retriever,
        tokenizer=tokenizer,
        es=es,
        df=df,
    )
    return
