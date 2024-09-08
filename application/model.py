from typing import List
from pydantic import BaseModel


class Query(BaseModel):
    question: str


class QueryList(BaseModel):
    queries: List[Query]
