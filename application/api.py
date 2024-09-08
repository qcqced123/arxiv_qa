from fastapi import FastAPI
from model import QueryList
from typing import List, Dict
from configuration import CFG
from omegaconf import OmegaConf
from utils.util import sync_config
from fastapi.middleware.cors import CORSMiddleware
from utils import make_queries, make_templates, call_inference_api
from utils import initialize_es, initialize_retriever, initialize_generator

# initialize the FastAPI Module
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize the configuration module
cfg = CFG
config_path = f'../config/inference/microsoft_e5_phi3.5.json'
sync_config(
    cfg,
    OmegaConf.load(config_path)
)

# initialize the necessary modules and post them to GPU
retriever_dict = initialize_retriever(cfg)  # get retriever module
generator_dict = initialize_generator(cfg)  # get generator module

# initialize the Elastic Search module for finding candidates document to answering the questions from users
es = initialize_es()


@app.post("/generate-answers/")
async def interface_fn(queries: QueryList) -> Dict:
    """ interface function for answering the questions from web users

    Args:
        queries (QueryList): List of queries from web page users

    Return:
        Dictionary of response(answer) to user's queries
    """
    query_list = make_queries(
        queries=queries
    )

    answer_list = await call_inference_api(
        cfg=cfg,
        retriever_dict=retriever_dict,
        generator_dict=generator_dict,
        es=es,
        queries=query_list,
    )

    templates = make_templates(
        queries=query_list,
        answers=answer_list
    )

    return {
        "responses": templates
    }
