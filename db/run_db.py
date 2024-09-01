import os
import subprocess
import pandas as pd
import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Dict
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from db.index_mapping import indexMapping
from db.helper import get_config, get_tokenizer, get_qlora_model

load_dotenv()


def create_index(es: Elasticsearch) -> None:
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


def encode_text(encoder: nn.Module, text: str) -> Tensor:
    """ function for extracting the embedding of documents

    Args:
        encoder (nn.Module): embedding mapper of input texts
        text: str, text for encoding

    return:
        Tensor, encoded text
    """
    return encoder.encode(text, show_progress_bar=True)


def encode_docs(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """ function for encoding documents

    Args:
        df: pd.DataFrame, dataframe containing [paper_id, doc_id, doc, doc embedding]
        encoder: SentenceTransformer, embedding mapper for encoding

    return:
        pd.DataFrame, dataframe containing [paper id, doc id, title, doc, doc embedding]
    """
    df['DocEmbedding'] = df['doc'].apply(lambda x: encode_text(x, encoder))
    return df


def search_candidates(query: str, encoder, es: Elasticsearch, top_k: int = 5, candidates: int = 500) -> List[Dict]:
    """ function for semantic searching with input queries, finding best matched candidates in elastic search engine

    Args:
        query: str, input query for searching
        encoder: SentenceTransformer, encoder for encoding text
        es: Elasticsearch, elastic search engine
        top_k: int, number of top k candidates for searching
        candidates: int, number of candidates for searching
    """

    h = encode_text(query, encoder)
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


def insert_doc_embedding(encoder: nn.Module, es: Elasticsearch, df: pd.DataFrame) -> None:
    """ function for inserting doc embedding into elastic search engine

    Args:
        encoder (nn.Module):
        es: Elasticsearch, elastic search engine
        df: pd.DataFrame, dataframe containing [paper id, doc id, doc, doc embedding]
    """
    df = encode_docs(df, encoder)
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

