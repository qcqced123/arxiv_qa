import os
import subprocess
import pandas as pd

from torch import Tensor
from typing import List, Dict
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from db.index_mapping import indexMapping

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


def get_encoder(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> SentenceTransformer:
    """ function for getting encoder

    Args:
        model_name: str, model name for the encoder

    return:
        SentenceTransformer, encoder object
    """
    return SentenceTransformer(model_name)


def encode_text(text: str, encoder: SentenceTransformer) -> Tensor:
    """ function for encoding

    Args:
        text: str, text for encoding
        encoder: SentenceTransformer, embedding mapper for encoding

    return:
        Tensor, encoded text
    """
    return encoder.encode(text, show_progress_bar=True)


def encode_docs(df: pd.DataFrame, encoder: SentenceTransformer) -> pd.DataFrame:
    """ function for encoding documents

    Args:
        df: pd.DataFrame, dataframe containing [paper_id, doc_id, doc, doc embedding]
        encoder: SentenceTransformer, embedding mapper for encoding

    return:
        pd.DataFrame, dataframe containing [paper id, doc id, title, doc, doc embedding]
    """
    df['DocEmbedding'] = df['doc'].apply(lambda x: encode_text(x, encoder))
    return df


def search_candidates(query: str, encoder: SentenceTransformer, es: Elasticsearch, top_k: int = 5, candidates: int = 500) -> List[Dict]:
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


def insert_doc_embedding(encoder: SentenceTransformer, es: Elasticsearch, df: pd.DataFrame) -> None:
    """ function for inserting doc embedding into elastic search engine

    Args:
        encoder: SentenceTransformer, embedding mapper for encoding
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

