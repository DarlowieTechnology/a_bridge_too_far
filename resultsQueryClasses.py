from enum import Enum, unique

from typing import List, Dict, Any

from pydantic import BaseModel, Field


@unique
class SEARCH(Enum) :
    BM25S = "bm25s"
    SEMANTIC = "semantic"

#--------------------OneQueryAppResult-------------------------------------

class OneQueryAppResult(BaseModel):
    """represents one query result in query app"""
    identifier: str = Field(..., description="identifier of record")
    title: str = Field(..., description="title of record")
    report: str = Field(..., description="report document name")
    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")

#----------------------OneQueryResultList-----------------------------------

class OneQueryResultList(BaseModel):
    """represents collection of one query results"""
    result_dict: Dict[str, OneQueryAppResult] = Field(default=None, description="dict of one query results, key by issue identifier")
    query : Any = Field(..., description="query used in search")
    searchType : SEARCH = Field(..., description="type of search used")
    label : str = Field(..., description="unique label of search run")

    def appendResult(self, identifier : str, title : str, report : str, score : float, rank : int) :
        self.result_dict[identifier] = OneQueryAppResult(
            identifier = identifier,
            title = title,
            report = report,
            score = score,
            rank = rank
        )

#--------------------AllQueryResults-------------------------------------

class AllQueryResults(BaseModel):
    """represents collection of all query results"""
    result_lists: List[OneQueryResultList] = Field(default=None, description="List of dict - all result")
    rrfScores : Dict[str, List[tuple[int, OneQueryAppResult]]] = Field(..., description="RRF ranks of issues in descending order")
    overall_score : float = Field(default=0, description="Overall score of result quality")
