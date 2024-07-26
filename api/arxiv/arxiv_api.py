import arxiv
import pandas as pd

from typing import List
from tqdm.auto import tqdm
from multiprocessing import Pool
from dataset_class.preprocessing import load_all_types_dataset


def set_sorting(sorting: str = 'relevance') -> object:
    """ Set the sorting criterion for the search results.

    if you pass argument sorting example below:
        relevance: arxiv.SortCriterion.Relevance
        latest_date: arxiv.SortCriterion.SubmittedDate

    Args:
        sorting: default str, sorting criterion for the search results,
                 Possible values are: 'relevance', 'lastUpdatedDate', 'submittedDate'

    Returns:
        arxiv.SortCriterion: object, sorting criterion for the search results

    """
    if sorting == 'relevance':
        return arxiv.SortCriterion.Relevance

    elif sorting == 'submittedDate':
        return arxiv.SortCriterion.SubmittedDate

    elif sorting == 'lastUpdatedDate':
        return arxiv.SortCriterion.LastUpdatedDate


def main_loop(queries: List[str], data_type: str = 'insert', max_results: int = 10, sorting=arxiv.SortCriterion.Relevance) -> None:
    """ main loop function for downloading query output from arxiv

    this function will download the paper named change 2110.03353v1 into it's title

    Usage:
        queries: 'iclr2020', 'ELECTRA', 'NLP', 'Transformer' ...
        max_results: 10, 20, 30, 40, 50 ...
        sorting: 'relevance', 'submittedDate', 'lastUpdatedDate'

    NLP conference list:
        ACL, EMNLP, NAACL, COLING, EACL

    Args:
        queries: default List[str], query string for searching the arxiv database
        data_type: 'insert', 'test' args for determining the download file's name and path
        max_results: int, maximum number of results to return
        sorting: object, sorting criterion for the search results

    Returns:
        arxiv_df: pd.DataFrame, dataframe containing the search results
    """
    for query in tqdm(queries):
        try:
            client = arxiv.Client(page_size=50, delay_seconds=10, num_retries=3)
            result = client.results(
                arxiv.Search(query=query, max_results=max_results, sort_by=sorting)
            )

            paper_list = []
            for paper in tqdm(result):
                paper_list.append(paper)
                url = paper.entry_id
                title = paper.title.replace('/', '_')
                pid = query if data_type == 'insert' else url[url.find('abs/') + len('abs/'):][:-2]
                filename = f"{pid}_{title}.pdf"
                paper.download_pdf(
                    dirpath='./train/',
                    filename=filename
                )
        except Exception as e:
            print(f'Error: {e}')

    return


def remove_exist_paper_list():
    query = pd.read_csv('./paper_id_list.csv').paper_id.tolist()  # next time, start at 430~2500

    try:
        exist_list = load_all_types_dataset('./exist_list.pkl.pkl')
        query = list(set(query) - set(exist_list))

    except FileNotFoundError as e:
        pass

    return query


if __name__ == '__main__':
    # q = sys.stdin.readline().rstrip()
    # return_results = int(sys.stdin.readline())
    # standard = sys.stdin.readline().rstrip()
    standard = 'relevance'
    values = set_sorting(sorting=standard)

    n_jobs = 4
    query = remove_exist_paper_list()
    chunked = [query[i:i + len(query)//n_jobs] for i in range(0, len(query), len(query)//n_jobs)]
    resume_chunked = [chunk for chunk in chunked]

    with Pool(processes=n_jobs) as pool:
        pool.map(main_loop, resume_chunked)
