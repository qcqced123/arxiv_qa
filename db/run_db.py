import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch import Tensor
from tqdm.auto import tqdm
from typing import List, Dict
from configuration import CFG
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from db.index_mapping import indexMapping
from db.helper import get_config, get_tokenizer, get_qlora_model

from model.pooling import MeanPooling
from transformers import AutoTokenizer

load_dotenv()


def create_index(model_name: str, es: Elasticsearch) -> None:
    """ function for creating index in elastic search engine with index mapping object in index_mapping.py
    """
    try:
        es.indices.create(index="document_embedding", mappings=indexMapping)

    except Exception as e:
        print(f"Error Message: {e}")

    return


def delete_index(es: Elasticsearch) -> None:
    """ function for deleting index in elastic search engine
    """
    es.indices.delete(index="document_embedding")
    return


def get_encoder(model_name: str) -> nn.Module:
    """ function for getting encoder fr inserting question or document in local DB

    Args:
        model_name (str): path of encoder model from local disk, already fine-tuned with question-document pair dataset

    return:
        nn.Module
    """
    config = get_config(model_name)
    model = get_qlora_model(
        model_name=model_name,
        config=config,
        bit_config=None,
        device="cuda:0",
        model_dtype=torch.bfloat16
    )
    return model


# must insert pytorch no grad context manager
@torch.no_grad()
def encode_text(
    cfg,
    encoder: nn.Module,
    pooling: nn.Module,
    tokenizer: AutoTokenizer,
    text: str
) -> Tensor:
    """ function for extracting the embedding of documents

    Args:
        cfg: dict, configuration module
        encoder (nn.Module): embedding mapper of input texts
        pooling (nn.Module): module of pooling encoder's output
        tokenizer (AutoTokenizer): module of tokenizing the inputs
        text: str, text for encoding

    return:
        embedding tensor of input text
    """
    # tokenize the input text for embedding model
    inputs = tokenizer.encode_plus(
        text,
        max_length=cfg.max_len,
        truncation=True,
        return_tensors="pt"
    )

    for k, v in inputs.items():
        inputs[k] = v.to(cfg.device)

    # extract the input text's embedding tensor
    h = encoder(**inputs)
    features = h.last_hidden_state
    embed = pooling(
        last_hidden_state=features,
        attention_mask=inputs["attention_mask"]
    )
    return embed


def encode_docs(
    cfg: CFG,
    encoder: nn.Module,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame
) -> pd.DataFrame:
    """ function for encoding documents

    Args:
        cfg (CFG): configuration module
        encoder (nn.Module): embedding mapper of input texts
        tokenizer (AutoTokenizer): module of tokenizing the inputs
        df: pd.DataFrame, dataframe containing [paper_id, doc_id, doc, doc embedding]

    return:
        pd.DataFrame, dataframe containing [paper id, doc id, title, doc, doc embedding]
    """
    pooling = MeanPooling()
    df['DocEmbedding'] = np.array([encode_text(cfg, encoder, pooling, tokenizer, text).cpu().numpy() for text in tqdm(df["inputs"].tolist())])
    return df


def search_candidates(
    cfg: CFG,
    encoder: nn.Module,
    tokenizer: AutoTokenizer,
    query: str,
    es: Elasticsearch,
    top_k: int = 5,
    candidates: int = 500
) -> List[Dict]:
    """ function for semantic searching with input queries, finding best matched candidates in elastic search engine

    Args:
        cfg (CFG): configuration module
        encoder (nn.Module): embedding mapper of input texts
        tokenizer (AutoTokenizer): module of tokenizing the inputs
        query (str): input query for searching
        es (Elasticsearch): elastic search engine
        top_k (int): number of top k candidates
        candidates (int): number of candidates
    """
    pooling = MeanPooling()
    h = encode_text(
        cfg=cfg,
        encoder=encoder,
        pooling=pooling,
        tokenizer=tokenizer,
        text=query
    )
    query = {
        "field": "DocEmbedding",
        "query_vector": h,
        "k": top_k,
        "num_candidates": candidates
    }

    return_data = ["paper_id", "doc_id", "title", "doc"]
    candidate = es.knn_search(
        index="document_embedding",
        knn=query,
        source=return_data
    )
    return candidate['hits']['hits']


def insert_doc_embedding(
    cfg: CFG,
    encoder: nn.Module,
    tokenizer: AutoTokenizer,
    es: Elasticsearch,
    df: pd.DataFrame
) -> None:
    """ function for inserting doc embedding into elastic search engine

    Args:
        cfg (CFG): configuration module
        encoder (nn.Module):
        tokenizer (AutoTokenizer):
        es: Elasticsearch, elastic search engine
        df: pd.DataFrame, dataframe containing [paper id, doc id, doc, doc embedding]
    """
    df = encode_docs(
        cfg=cfg,
        encoder=encoder,
        tokenizer=tokenizer,
        df=df,
    )

    df.to_csv("document_embedding_arxiv.csv", index=False)
    records = df.to_dict(orient='records')
    try:
        for record in records:
            es.index(index="document_embedding", document=record, id=record['doc_id'])

        print("Document Embedding Inserted Successfully")

    except Exception as e:
        print("Error in inserting doc embedding:", e)

    return


def run_engine(url: str, auth: str, cert: str) -> Elasticsearch:
    """ function for running elastic engine

    Args:
        url: str, local host url for the elastic engine, default is "https://localhost:9200"
        auth: str, authentication for the elastic engine
        cert: str, certificate for the elastic engine

    return:
        es: Elasticsearch, elasticsearch engine object
    """
    es = None
    # command = f"{os.environ.get('LINUX_RUNNER_PATH')}"

    try:
        # subprocess.run(command, shell=True, check=True)
        es = Elasticsearch(hosts=url, basic_auth=("elastic", auth), ca_certs=cert)
        print(es.ping())

    except ConnectionError as e:
        print("Connection Error:", e)

    return es

