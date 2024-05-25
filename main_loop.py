import os
import torch
import argparse
import warnings
import platform

from omegaconf import OmegaConf
from dotenv import load_dotenv
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from huggingface_hub import login

from configuration import CFG
from trainer.trainer import TextGenerationTuner
from dataset_class.preprocessing import load_all_types_dataset
from dataset_class.make_dataset import build_doc_embedding_db
from db.run_db import run_engine, create_index, get_encoder, insert_doc_embedding, search_candidates

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


def login_to_huggingface() -> None:
    login(os.environ.get("HUGGINGFACE_API_KEY"))
    return


def get_db_url():
    return os.environ.get('ELASTIC_ENGINE_URL')


def get_db_auth(os_type: str) -> str:
    return os.environ.get('MAC_ELASTIC_ENGINE_PASSWORD') if os_type == "Darwin" else os.environ.get('LINUX_ELASTIC_ENGINE_PASSWORD')


def get_db_cert(os_type: str) -> str:
    return os.environ.get('MAC_CA_CERTS') if os_type == "Darwin" else os.environ.get('LINUX_CA_CERTS')


def main(cfg: CFG, train_type: str, model_config: str) -> None:
    """ main loop function for running the engine

    workflow:
        1) login to huggingface hub
        2) update the config file

        3) run the elasticsearch engine
            - create index (if not exists)
            - make doc embedding
            - insert doc embedding

        4) search the doc embedding

        5) run the text generation tuner for generating the answer for the input query
    """
    login_to_huggingface()
    config_path = f'config/{train_type}/{model_config}.json'
    sync_config(cfg, OmegaConf.load(config_path))

    os_type = platform.system()
    es = run_engine(url=get_db_url(), auth=get_db_auth(os_type), cert=get_db_cert(os_type))
    try:
        create_index(es)

    except Exception as e:
        print("Error in creating index:", e)
        pass

    retriever = get_encoder()

    if train_type == "train":
        df = build_doc_embedding_db(
            load_all_types_dataset('./dataset_class/arxiv_qa/train_paper_meta_db.csv')
        )

        insert_doc_embedding(
            df=df,
            encoder=retriever,
            es=es
        )

    query = "What is the self-attention mechanism in transformer?"
    result = search_candidates(
        query=query,
        encoder=retriever,
        es=es,
        top_k=2,
    )

    context = ''
    for i, res in enumerate(result):
        curr = f"Title {i+1}: " + res['_source']['title'] + "\n"
        curr += res['_source']['doc']
        context += curr
        if i+1 != len(result):
            context += "\n\n"

    tuner = TextGenerationTuner(
        cfg=cfg,
        generator=g,
        es=es
    )
    generator = tuner.model_setting()
    answer = tuner.inference(
        model=generator,
        query=query,
        context=context,
        max_length=cfg.max_len,
        strategy=cfg.strategy,
        penalty_alpha=cfg.penalty_alpha,
        num_beams=cfg.num_beams,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        length_penalty=cfg.length_penalty,
        do_sample=cfg.do_sample,
        use_cache=cfg.use_cache
    )
    print(answer)
    return


if __name__ == '__main__':
    config = CFG
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("train_type", type=str, help="Train Type Selection")  # train, inference
    parser.add_argument("model_config", type=str, help="Model config Selection")  # name of retriever-generator
    args = parser.parse_args()

    main(config, args.train_type, args.model_config)
