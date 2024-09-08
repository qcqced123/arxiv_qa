import os
import uvicorn

from pyngrok import ngrok
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from threading import Thread
from typing import List, Dict
from configuration import CFG
from dotenv import load_dotenv
from omegaconf import OmegaConf
from utils.util import sync_config
from application.model import QueryList
from fastapi.middleware.cors import CORSMiddleware
from application.utils import make_queries, make_templates, call_inference_api
from application.utils import initialize_es, initialize_retriever, initialize_generator

load_dotenv()
ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

# initialize the FastAPI Module
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# initialize the ngrok tunnel
# run the ngrok tunnel in the background for getting public url
public_url = None
def start_ngrok():
    global public_url
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
    print(f" * Ngrok tunnel available at: {tunnel}")
    print(f" * You can access your app at {tunnel}/")

@app.get("/ngrok-url")
async def get_ngrok_url():
    if public_url is None:
        return JSONResponse(content={"error": "Ngrok tunnel not started yet"})

    return JSONResponse(content={"url": public_url})

Thread(target=start_ngrok).start()


# initialize the configuration module
cfg = CFG
config_path = f'config/inference/microsoft_e5_phi3.5.json'
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

    answer_list = call_inference_api(
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

if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print(f" * Ngrok tunnel available at: {public_url}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
