from typing import List, Dict
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from db.run_db import search_candidates


def build_context(result: List[Dict]) -> str:
    """ function for building context text for passing the context input to the model """
    context = ''
    for i, res in enumerate(result):
        curr = f"Title {i + 1}: " + res['_source']['title'] + "\n"
        curr += res['_source']['doc']
        context += curr
        if i + 1 != len(result):
            context += "\n\n"

    return context


def query_encoder(
    es: Elasticsearch,
    retriever: SentenceTransformer,
    query: str = "Hello, Claude!",
    top_k=5
):
    """ function for encoding input query and searching the nearest top-k documents from document embedding db

    Args:
        es: Elasticsearch, elastic search engine
        retriever: SentenceTransformer, encoder for encoding text
        query: str, input query for searching
        top_k: int, number of top k candidates for searching

    """
    result = search_candidates(query, retriever, es, top_k=top_k)
    for i, res in enumerate(result):  # print the result of retrieved documents
        print(f"product {i + 1}: {res}")

    return build_context(result)
