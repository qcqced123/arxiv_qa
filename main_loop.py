import os
import torch
import argparse
import warnings
import platform
import pandas as pd

from typing import List
from tqdm.auto import tqdm
from dotenv import load_dotenv
from omegaconf import OmegaConf
from multiprocessing import pool
from huggingface_hub import login

from utils.util import sync_config
from utils.helper import check_library, all_type_seed

from configuration import CFG
from trainer.train_loop import train_loop, inference_loop
from document_encoder.document_encoder import document_encoder
from db.run_db import run_engine, create_index, get_encoder, insert_doc_embedding, search_candidates
from dataset_class.text_chunk import chunk_by_length, chunk_by_recursive_search, cut_pdf_to_sub_module_with_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"

load_dotenv()
check_library(True)
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
all_type_seed(CFG, True)

g = torch.Generator()
g.manual_seed(CFG.seed)

# global variables
base_path = r"api/arxiv/train/"


def login_to_huggingface() -> None:
    login(os.environ.get("HUGGINGFACE_API_KEY"))
    return


def get_db_url():
    return os.environ.get('ELASTIC_ENGINE_URL')


def get_db_auth(os_type: str) -> str:
    return os.environ.get('MAC_ELASTIC_ENGINE_PASSWORD') if os_type == "Darwin" else os.environ.get('LINUX_ELASTIC_ENGINE_PASSWORD')


def get_db_cert(os_type: str) -> str:
    return os.environ.get('MAC_CA_CERTS') if os_type == "Darwin" else os.environ.get('LINUX_CA_CERTS')


def make_loop(path_list: List[str]) -> None:
    for path in tqdm(path_list):
        pid, title = path.split('_')  # current pdf file's paper id and title for making dataframe
        result = cut_pdf_to_sub_module_with_text(
            path=base_path + path,
            strategy="hi_res",
            model_name="yolox"
        )
        document = chunk_by_recursive_search(
            text=result['text'],
            chunk_size=1024,
            over_lapping_size=128
        )
        documents = [e.page_content for e in document]
        for e in result["table"]:
            documents.append(e)

        for e in result["formula"]:
            documents.append(e)
        documents.append(result['reference'])

        data_list = [[pid, f"{pid}_{i}", title, d] for i, d in enumerate(documents)]
        curr = pd.DataFrame(
            data=data_list,
            columns=['paper_id', 'doc_id', 'title', 'doc']
        )
        output_path = f"dataset_class/datafolder/arxiv_qa/partition/{pid}.csv"
        curr.to_csv(output_path, index=False)

    return


def main(cfg: CFG, pipeline_type: str, model_config: str) -> None:
    """ main loop function for running the engine

    workflow:
        1) login to huggingface hub
        2) update the config file

        3) run the elasticsearch engine

        # CLI argument signature: "make"
        # this branch will be called when the pipeline type var is set to "make"
        4) make the document embedding db, question-document dataset for fine-tuning query encoder with metric learning
            - run the MySQL DB Server for saving the document DataFrame
            - load the pdf file from pdf db

            - apply text chunking strategy from text_chunk module
                => this process will be done by multi-processing

            - make the document dataframe for document embedding db
            - generate the question data by using Google Gemini API, Llama3-8b model
            - insert final document dataframe into the MySQL DB Server

        # CLI argument signature: "insert"
        # this branch will be called when the pipeline type var is set to "insert"
        5) run document encoder
            - create the db index (if not exists)
            - get the document dataframe from the MySQL DB Server (document db)
            - make the document embedding db
            - insert the document embedding into the elastic search engine

        # CLI argument signature: "fine_tune", "inference"
        # this branch will be called when the pipeline type var is set to "fine_tune" or "inference"
        6) run query encoder
            - get query from the user or query db
            - encode or project the query into embedding space
            - search the candidates from the document embedding db

        7) run the text generation tuner for generating the answer for the input query

    """
    login_to_huggingface()
    config_path = f'config/{pipeline_type}/{model_config}.json'
    sync_config(cfg, OmegaConf.load(config_path))

    os_type = platform.system()
    es = run_engine(url=get_db_url(), auth=get_db_auth(os_type), cert=get_db_cert(os_type))

    # need to abstract this branch
    # branch for calling pipeline that builds the document embedding db, q-doc dataset
    if pipeline_type == "make":
        base_path = r"api/arxiv/train/"
        path_list = os.listdir(base_path)

        # apply text chunking strategy from text_chunk module
        # linear for-loop by pdf list
        # multi-processing for-loop by pdf list
        df = pd.DataFrame(columns=['paper_id', 'doc_id', 'title', 'doc'])
        size = len(path_list)//cfg.n_jobs
        chunked = [path_list[i:i+size] for i in range(0, len(path_list), size)]

        for path in tqdm(path_list):
            pid, title = path.split('_')  # current pdf file's paper id and title for making dataframe
            result = cut_pdf_to_sub_module_with_text(
                path=base_path+path,
                strategy="hi_res",
                model_name="yolox"
            )
            document = chunk_by_recursive_search(
                text=result['text'],
                chunk_size=1024,
                over_lapping_size=128
            )
            documents = [e.page_content for e in document]
            for e in result["table"]:
                documents.append(e)

            for e in result["formula"]:
                documents.append(e)
            documents.append(result['reference'])

            # linear for-loop by elements of pdf
            data_list = [[pid, f"{pid}_{i}", title, d] for i, d in enumerate(documents)]
            curr = pd.DataFrame(
                data=data_list,
                columns=['paper_id', 'doc_id', 'title', 'doc']
            )
            df = pd.concat([df, curr], axis=0)

            # backup
            output_path = f"dataset_class/datafolder/arxiv_qa/partition/{pid}.csv"
            curr.to_csv(output_path, index=False)

        # backup
        output_path = f"dataset_class/datafolder/arxiv_qa/total/arxiv_paper_document_db.csv"
        df.to_csv(output_path, index=False)

        # need to add generate question pipeline

    # branch for calling pipeline that inserts the document embedding into the elastic search engine
    elif pipeline_type == "insert":
        retriever = get_encoder()
        df = pd.read_csv('dataset_class/datafolder/arxiv_qa/total/arxiv_paper_document_db.csv')
        document_encoder(
            es=es,
            retriever=retriever,
            df=df
        )

    # branch for calling pipeline that fine-tune the query encoder with metric learning, generator with clm
    elif pipeline_type == "fine_tune":
        # train_loop()
        pass

    # branch for calling pipeline that generates the best answer for the input query
    elif pipeline_type == "inference":
        answers = inference_loop(
            cfg=cfg,
            pipeline_type=pipeline_type,
            model_config=model_config,
            es=es
        )
        for i, answer in enumerate(answers):
            print(f"{i+1}-th question's answer is: \n\n {answer}")
    return


if __name__ == '__main__':
    config = CFG
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("pipeline_type", type=str, help="Train Type Selection")  # train, inference
    parser.add_argument("model_config", type=str, help="Model config Selection")  # name of retriever-generator
    args = parser.parse_args()

    main(config, args.pipeline_type, args.model_config)
