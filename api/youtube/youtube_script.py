import re, os
import pandas as pd

import openai
import google.generativeai as genai

from typing import List
from dotenv import load_dotenv
from api.youtube.language_setting import LANGUAGE
from youtube_transcript_api import YouTubeTranscriptApi


load_dotenv()


def youtube_script_api(url_path: str) -> str:
    """ extract the script from the given YouTube video URL

    Args:
        url_path (str): The YouTube video URL for extracting the script
    """
    output = ''
    try:
        video_id = url_path.replace('https://www.youtube.com/watch?v=', '')
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=LANGUAGE)
        for x in transcript:
            sentence = x['text']
            output += f'{sentence}\n'

    except Exception as e:
        print(e)

    return output


if __name__ == "__main__":
    url = 'https://www.youtube.com/watch?v=b_2v9Hpfnbw&ab_channel=NicholasBroad'

    text = youtube_script_api(url)
    print(f"YouTube Script: {text}")
