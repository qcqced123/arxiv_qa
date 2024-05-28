import arxiv
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool


def set_sorting(sorting: str = 'relevance') -> object:
    """
    Set the sorting criterion for the search results.

    if you pass argument sorting example below:
        relevance: arxiv.SortCriterion.Relevance
        latest_date: arxiv.SortCriterion.SubmittedDate

    Args:
        sorting: default str, sorting criterion for the search results,
                 Possible values are: 'relevance', 'lastUpdatedDate', 'submittedDate'

    Returns:

    """
    if sorting == 'relevance':
        return arxiv.SortCriterion.Relevance

    elif sorting == 'submittedDate':
        return arxiv.SortCriterion.SubmittedDate

    elif sorting == 'lastUpdatedDate':
        return arxiv.SortCriterion.LastUpdatedDate


def main_loop(query: str, data_type: str = 'train', max_results: int = 10, sorting=arxiv.SortCriterion.Relevance):
    """ main loop function for downloading query output from arxiv

    this function will download the paper named change 2110.03353v1 into it's title
    Usage:
        query: 'iclr2020', 'ELECTRA', 'NLP', 'Transformer' ...
        max_results: 10, 20, 30, 40, 50 ...
        sorting: 'relevance', 'submittedDate', 'lastUpdatedDate'

    NLP conference list:
        ACL, EMNLP, NAACL, COLING, EACL

    Args:
        query: str, query string for searching the arxiv database
        data_type: 'train', 'test' args for determining the download file's name and path
        max_results: int, maximum number of results to return
        sorting: object, sorting criterion for the search results

    Returns:
        arxiv_df: pd.DataFrame, dataframe containing the search results
    """

    client = arxiv.Client(page_size=1000, delay_seconds=2, num_retries=3)
    result = client.results(
        arxiv.Search(query=query, max_results=max_results, sort_by=sorting)
    )

    paper_list = []
    for paper in tqdm(result):
        paper_list.append(paper)
        url = paper.entry_id
        title = paper.title.replace('/', '_')
        pid = query if data_type == 'train' else url[url.find('abs/') + len('abs/'):][:-2]
        filename = f"{pid}_{title}.pdf"
        paper.download_pdf(
            dirpath='./download/train/',
            filename=filename
        )

    arxiv_df = pd.DataFrame([vars(paper) for paper in paper_list])
    return arxiv_df


if __name__ == '__main__':
    # q = sys.stdin.readline().rstrip()
    # return_results = int(sys.stdin.readline())
    # standard = sys.stdin.readline().rstrip()
    standard = 'relevance'
    values = set_sorting(sorting=standard)

    query = pd.read_csv('./download/paper_id_list.csv').paper_id.tolist()

    with Pool(processes=4) as pool:
        results = pool.map(main_loop, query[0:8000])  # 0~8000 for train, 8000~ for test
