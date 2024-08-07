from typing import List, Dict
from elasticsearch import Elasticsearch
from db.run_db import search_candidates
from sentence_transformers import SentenceTransformer


def build_context(query: str, result: List[Dict]) -> str:
    """ function for building context text for passing the context input to the model

    Args:
        query: str, input query for searching for adding inputs of the generator
        result: List[Dict], list of dictionary object for the search result

    Returns:
        input text (context text) for the generator
    """

    context = f"Given the following context about arxiv paper, especially in the field of computer science:\n\n"
    for i, res in enumerate(result):
        curr = f"title{i+1}: {res['_source']['title']}\ncontext{i+1}: {res['_source']['doc']}"
        context += curr
        if i + 1 != len(result):
            context += "\n\n"

    context += query
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
    result = search_candidates(
        query,
        retriever,
        es,
        top_k=top_k
    )
    return build_context(query, result)
