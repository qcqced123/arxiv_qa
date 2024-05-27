import os
import pandas as pd
import googleapiclient.discovery

from typing import List
from tqdm.auto import tqdm
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

credentials = service_account.Credentials.from_service_account_file(
    f"{os.environ.get('GOOGLE_CLOUDE_CONSOLE_CREDENTIAL_PATH')}"
)


def youtube_search_api(query: str, keyword: str = 'review', order: str = 'relevance', nums: int = 10, src: str = None, region: str = 'US') -> List[str]:
    """ helper function for searching YouTube videos by using the given query

    Args:
        query (str): The search query for finding YouTube videos, in this case, you pass the the product name
        keyword (str): The search keyword for finding YouTube videos, example 'review', 'unboxing', 'tutorial' ...
        order (str): default 'relevance', the order of the search results
                     (relevance, date, rating, viewCount, title, videoCount)
        nums (int): The number of search results to return
        src (str): The start datetime of result of search query
        region (str): default 'US', the region code for the search query, setting watchable region

    References:
        https://developers.google.com/youtube/v3/determine_quota_cost?hl=ko
        https://developers.google.com/youtube/v3/getting-started?hl=ko#quota
        https://developers.google.com/youtube/v3/docs/search/list?apix=true&apix_params=%7B%22part%22%3A%22snippet%22%2C%22q%22%3A%22marine%22%2C%22relevanceLanguage%22%3A%22fr%22%7D&hl=ko

    """
    q = query + ' ' + keyword
    youtube = googleapiclient.discovery.build(
        serviceName='youtube',
        version='v3',
        developerKey=os.environ.get('YOUTUBE_API_KEY'),
        # credentials=credentials
    )

    request = youtube.search().list(
        part="snippet",
        q=q,
        type="video",
        order=order,
        maxResults=nums,
        publishedAfter=src,
        regionCode=region
    )

    response = request.execute()
    title, description, video_id = None, None, None
    try:
        result = response['items'][0]
        title = result['snippet']['title']
        description = result['snippet']['description']
        video_id = f"https://www.youtube.com/watch?v={result['id']['videoId']}"

    except Exception as e:
        print(e)
        pass

    return [query, title, description, video_id]


if __name__ == '__main__':
    task_keyword = "review"
    db = pd.read_csv("../instacart/insta_lidi_target_walmart.csv")
    df = pd.DataFrame(
        [youtube_search_api(query, task_keyword) for query in tqdm(db['product_name'].unique())],
        columns=['product_name', 'video_title', 'video_description', 'video_url']
    )
    df.to_csv(f"youtube_video_list_for_product_info.csv", index=False)
    print(df)
