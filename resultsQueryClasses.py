from enum import Enum, unique

from typing import Generic, List, Dict, Literal, Union, TypeVar

from pydantic import BaseModel, Field


@unique
class SEARCH(str, Enum) :
    NONE = "none"
    BM25S = "bm25s"
    SEMANTIC = "semantic"

#--------------------one query result-------------------------------------

class OneQueryChunkResult(BaseModel):
    """represents one query result in Discovery app"""

    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")
    chunk: str = Field( default = "", description="chunk of text in document")
    chunkID : int = Field( default = -1, description="chunk id in document")
    document: str = Field( default = "", description="document name")
    searchTypeName : str = Field( default=SEARCH.NONE.value , description="name of search type used")


class OneIndexerQueryResult(BaseModel):
    """represents one query result in Indexer app"""
    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")
    identifier : str = Field(..., description="identifier of record")
    title : str = Field(..., description="title of record")
    report : str = Field(..., description="report document name")


#----------------------OneQueryResultList-----------------------------------

class OneChunkQueryResultList(BaseModel):
    """represents collection of one query results for Discovery app"""
    label : str = Field( "", description="unique label of search run")
    query : list[str] = Field(default = [], description="query used in search")
    result_dict: Dict[str, OneQueryChunkResult] = Field(default = {}, description="dict, key is issue identifier, value is list of results from one query")

    def appendQueryResult(self, identifier: str, queryResult : OneQueryChunkResult) :
        self.result_dict[identifier] = queryResult


class OneIndexerQueryResultList(BaseModel):
    """represents collection of one query results for Discovery app"""
    label : str = Field( "", description="unique label of search run")
    query : list[str] = Field(default = [], description="query used in search")
    result_dict: Dict[str, OneIndexerQueryResult] = Field(default = {}, description="dict, key is issue identifier, value is list of results from one query")

    def appendQueryResult(self, identifier: str, queryResult : OneIndexerQueryResult) :
        self.result_dict[identifier] = queryResult


#--------------------AllQueryResults-------------------------------------

class IdentifierQueryResults(BaseModel):
    """represents unique identifier, RRF score, outlier info for one item"""
    identifier: str = Field(default = "", description="unique identifier")
    rrfRank : float = Field(default = 0.0, description="RRF rank")
    outlierIQR: bool = Field( default = False, description="outlier by IQR")
    outlierZScore: bool = Field( default = False, description="outlier by Z Score")


class RRFScores(BaseModel):
    """represents all RRF scores"""
    scoresDict : Dict[str, IdentifierQueryResults] = Field(default={}, description="RRF ranks in descending order")


class AllChunkQueryResults(BaseModel):
    """represents collection of all query results for Discovery"""
    query : list[str] = Field(default = [], description="Original query as a list of strings")
    rrfScores : RRFScores = Field(default=None, description="RRF ranks in descending order")
    listQueryResults: List[ OneChunkQueryResultList ] = Field(default=None, description="List of OneChunkQueryResultList")


class AllIndexerQueryResults(BaseModel):
    """represents collection of all query results for Indexer and Query"""
    query : list[str] = Field(default = [], description="Original query")
    rrfScores : RRFScores = Field(default=None, description="RRF ranks in descending order")
    listQueryResults: List[ OneIndexerQueryResultList ] = Field(default=None, description="List of OneIndexerQueryResultList")
