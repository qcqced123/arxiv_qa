import pandas as pd

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from db.run_db import create_index, insert_doc_embedding


def document_encoder(retriever: SentenceTransformer, es: Elasticsearch, df: pd.DataFrame) -> None:
    """ function for creating, inserting doc embedding into elastic search engine """
    try:
        create_index(es)

    except Exception as e:
        print(f"Error: {e}")

    insert_doc_embedding(
        encoder=retriever,
        es=es,
        df=df,
    )
    return
