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
from transformers import logging as transformers_logging

from utils.util import sync_config
from utils.helper import check_library, all_type_seed

from db.helper import get_tokenizer
from db.run_db import run_engine, get_encoder
from db.run_db import login_to_huggingface, get_db_url, get_db_auth, get_db_cert

from configuration import CFG
from document_encoder.document_encoder import document_encoder
from trainer.train_loop import train_loop, generate_loop, inference_loop
from dataset_class.preprocessing import jump_exist_paper, merge_partition_files
from dataset_class.text_chunk import chunk_by_length, chunk_by_recursive_search, cut_pdf_to_sub_module_with_text

from inference.helper import init_normalizer
from inference.vllm_inference import initialize_llm, get_sampling_params

os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


load_dotenv()
check_library(False)
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

all_type_seed(CFG, True)
g = torch.Generator()
g.manual_seed(CFG.seed)

# global variables
base_path = r"api/arxiv/train/"


def make_loop(path_list: List[str]) -> List:
    data_list = []
    for path in tqdm(path_list):
        print(path.split('_'))
        pid, title = path.split('_')  # current pdf file's paper id and title for making dataframe

        # jump logic if the current pdf file is already processed
        if jump_exist_paper(pid):
            continue

        try:
            result = cut_pdf_to_sub_module_with_text(
                path=base_path + path,
                strategy="hi_res",
                model_name="yolox"
            )
            document = chunk_by_recursive_search(
                text=result['text'],
                chunk_size=4096,
                over_lapping_size=512
            )
            documents = [e.page_content for e in document]
            for e in result["table"]:
                documents.append(e)

            for e in result["formula"]:
                documents.append(e)
            documents.append(result['reference'])

            data = [[pid, f"{pid}_{i}", title, d] for i, d in enumerate(documents)]
            data_list.extend(data)

            curr = pd.DataFrame(
                data=data,
                columns=['paper_id', 'doc_id', 'title', 'doc']
            )
            output_path = f"dataset_class/datafolder/arxiv_qa/partition/{pid}.csv"
            curr.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error Message: {e}")

    return data_list


def main(cfg: CFG, pipeline_type: str, model_config: str) -> None:
    """ interface function of main application (main loop function)
    you can choose the pipeline type by setting the pipeline type var

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
            - generate the question data by using any generator llm such as
              (microsoft/Phi-3-mini-128k-instruct)

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
    # start process lifecycle
    # login_to_huggingface()
    config_path = f'config/{pipeline_type}/{model_config}.json'
    sync_config(cfg, OmegaConf.load(config_path))

    os_type = platform.system()
    es = run_engine(url=get_db_url(), auth=get_db_auth(os_type), cert=get_db_cert(os_type))

    # need to abstract this branch
    # branch for calling pipeline that builds the document embedding db, q-doc dataset
    if pipeline_type == "make":
        # branch of getting elements from pdf files and making the document dataframe
        # if configuration var state(cfg.work_flow_state) is "init", then make the document dataframe
        # if configuration var state(cfg.work_flow_state) is "resume", then load the document dataframe
        df = None
        if cfg.work_flow_state == "init":
            path_list = os.listdir(base_path)

            # apply text chunking strategy from text_chunk module
            # linear for-loop by pdf list
            # multi-processing for-loop by pdf list
            size = len(path_list)//cfg.n_jobs
            chunked = [path_list[i:i+size] for i in range(0, len(path_list), size)]

            with pool.Pool(processes=cfg.n_jobs) as p:
                p.map(make_loop, chunked)

            output_path = f"dataset_class/datafolder/arxiv_qa/total/arxiv_paper_document_db.csv"
            df = merge_partition_files()
            df.to_csv(output_path, index=False)

        elif cfg.work_flow_state == "resume":
            df = pd.read_csv('dataset_class/datafolder/arxiv_qa/total/sampling_metric_learning_total_paper_chunk.csv')

        # branch of question generation pipeline
        # current generative API is vllm backend
        # default setting for foundation model is microsoft/Phi-3.5-mini-instruct with AWQ (4bit quantization)
        generator_dict = {
            "generator_tokenizer": get_tokenizer(cfg.generator_name),
            "generator": initialize_llm(cfg=cfg),
            "sampling_params": get_sampling_params(cfg),
            "text_normalizer": init_normalizer(mode="lower_cased", language="en")
        }
        generate_loop(
            cfg=cfg,
            flow=cfg.work_flow_state,
            df=df,
            generator_dict=generator_dict
        )

    # branch for calling pipeline that inserts the document embedding into the elastic search engine
    # you can select the document encoder model in configuration json file
    elif pipeline_type == "insert":
        retriever = get_encoder(cfg.model_name)  # pass your encoder model's path
        tokenizer = get_tokenizer(cfg.model_name)
        df = pd.read_csv('dataset_class/datafolder/arxiv_qa/arxiv_question_document_pair.csv')
        document_encoder(
            cfg=cfg,
            retriever=retriever,
            tokenizer=tokenizer,
            es=es,
            df=df
        )

    # branch for calling pipeline that fine-tune the query(~= document) encoder with metric learning
    # query and document encoder will be used same module
    elif pipeline_type == "fine_tune":
        train_loop(
            cfg=cfg,
            pipeline_type=pipeline_type,
            model_config=model_config
        )

    # branch for calling pipeline that generates the best answer for the input query
    elif pipeline_type == "inference":
        # make the dictionary for each module
        # load the retriever, generator and transfer to GPU
        retriever_dict = {
            "retriever_tokenizer": get_tokenizer(cfg.retriever_name),
            "retriever": get_encoder(cfg.retriever_name).to(cfg.device)
        }
        generator_dict = {
            "generator_tokenizer": get_tokenizer(cfg.generator_name),
            "generator": initialize_llm(cfg=cfg),
            "sampling_params": get_sampling_params(cfg)
        }
        answers = inference_loop(
            cfg=cfg,
            retriever_dict=retriever_dict,
            generator_dict=generator_dict,
            es=es
        )
        for i, answer in enumerate(answers):
            print(f"{i+1}-th question's answer is: \n\n {answer}")

    return


if __name__ == '__main__':
    config = CFG
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("pipeline_type", type=str, help="Train Type Selection")  # insert, inference
    parser.add_argument("model_config", type=str, help="Model config Selection")  # name of retriever-generator
    args = parser.parse_args()

    main(config, args.pipeline_type, args.model_config)
