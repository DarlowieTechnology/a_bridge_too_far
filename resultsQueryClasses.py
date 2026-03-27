from enum import Enum, unique

from typing import List, Dict, Any

from pydantic import BaseModel, Field


@unique
class SEARCH(str, Enum) :
    BM25S = "bm25s"
    SEMANTIC = "semantic"

#--------------------one query result-------------------------------------

class OneQueryBaseResult(BaseModel):
    """represents base class for query result"""
    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")


class OneQueryChunkResult(OneQueryBaseResult):
    """represents one query result in chunked document query"""
    chunk: str = Field(..., description="chunk text in document")
    chunkID : int = Field(..., description="chunk id in document")
    document: str = Field(..., description="document name")


class OneQueryAppResult(OneQueryBaseResult):
    """represents one query result in query app"""
    identifier: str = Field(..., description="identifier of record")
    title: str = Field(..., description="title of record")

#----------------------OneQueryResultList-----------------------------------

class OneQueryResultList(BaseModel):
    """represents collection of one query results"""
    result_dict: Dict[str, OneQueryChunkResult] = Field(default=None, description="dict of one query results, key by issue identifier")
    query : Any = Field(..., description="query used in search")
    searchType : SEARCH = Field(..., description="type of search used")
    label : str = Field(..., description="unique label of search run")


    def appendQueryAppResult(self, identifier : str, title : str, report : str, score : float, rank : int) :
        self.result_dict[identifier] = OneQueryAppResult(
            identifier = identifier,
            title = title,
            document = report,
            score = score,
            rank = rank
        )


    def appendQueryChunkResult(self, chunk : str, chunkID: int, document : str, score : float, rank : int) :
        identifier = document + "--" + str(chunkID)
        self.result_dict[identifier] = OneQueryChunkResult(
            chunk = chunk,
            chunkID = chunkID,
            document = document,
            score = score,
            rank = rank
        )
    


#--------------------AllQueryResults-------------------------------------

class IdentifierQueryResults(BaseModel):
    """represents unique identifier and list of query results from multiple queries"""
    identifier: str = Field(default = "", description="unique identifier")
    all_query_results: List[OneQueryChunkResult] = Field(default=None, description="List of query results for the identifier")


class RRFScores(BaseModel):
    """represents all RRF scores. Scores are rounded to 6 digits"""
    scoresDict : Dict[float, IdentifierQueryResults] = Field(..., description="RRF ranks in descending order")


class AllQueryResults(BaseModel):
    """represents collection of all query results"""
    result_lists: List[OneQueryResultList] = Field(default=None, description="List of OneQueryResultList - all result")
    rrfScores : RRFScores = Field(default=None, description="RRF ranks in descending order")
