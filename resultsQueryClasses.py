from enum import Enum, unique

from typing import List, Dict, Literal, Union

from pydantic import BaseModel, Field


@unique
class SEARCH(str, Enum) :
    NONE = "none"
    BM25S = "bm25s"
    SEMANTIC = "semantic"

#--------------------one query result-------------------------------------

class OneQueryBaseResult(BaseModel):
    """represents base class for query result"""
    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")


class OneQueryChunkResult(BaseModel):
    """represents one query result in chunked document query"""

    score : float = Field(..., description="distance/score of record")
    rank : int = Field(..., description="rank of the record in search")
    chunk: str = Field( default = "", description="chunk of text in document")
    chunkID : int = Field( default = -1, description="chunk id in document")
    document: str = Field( default = "", description="document name")
    searchTypeName : str = Field( default=SEARCH.NONE.value , description="name of search type used")


class OneQueryAppResult(OneQueryBaseResult):
    """represents one query result in query app"""
    identifier: str = Field(..., description="identifier of record")
    title: str = Field(..., description="title of record")

#----------------------OneQueryResultList-----------------------------------

class QuerySemantic(BaseModel):
    query : list[str] = Field(default = [], description="semantic query is a list of strings")
    searchType : Literal[SEARCH.SEMANTIC]


class QueryBM25s(BaseModel):
    query : list[list[str]] = Field(default = [], description="bm25s query is a list of list of strings")
    searchType : Literal[SEARCH.BM25S]


class OneQueryResultList(BaseModel):
    """represents collection of one query results"""
    result_dict: Dict[str, OneQueryChunkResult] = Field(default = {}, description="dict of one query results, key by issue identifier")
    query : Union[ QuerySemantic, QueryBM25s ] = Field(default = [], description="query used in search", discriminator='searchType')
    label : str = Field( "", description="unique label of search run")


    def appendQueryAppResult(self, identifier : str, title : str, report : str, score : float, rank : int) :
        self.result_dict[identifier] = OneQueryAppResult(
            identifier = identifier,
            title = title,
            document = report,
            score = score,
            rank = rank
        )


    def appendQueryChunkResult(self, score : float, rank : int, chunk : str, chunkID: int, document : str, searchTypeName : str) :
        identifier = document + "--" + str(chunkID)
        self.result_dict[identifier] = OneQueryChunkResult(
            score = score,
            rank = rank,
            chunk = chunk,
            chunkID = chunkID,
            document = document,
            searchTypeName = searchTypeName
        )
    


#--------------------AllQueryResults-------------------------------------

class IdentifierQueryResults(BaseModel):
    """represents unique identifier and list of query results from multiple queries"""
    identifier: str = Field(default = "", description="unique identifier")
    score : float = Field(default = 0.0, description="RRF score")
    chunk: str = Field( default = "", description="chunk of text in document")


class RRFScores(BaseModel):
    """represents all RRF scores"""
    scoresDict : Dict[str, IdentifierQueryResults] = Field(..., description="RRF ranks in descending order")


class AllQueryResults(BaseModel):
    """represents collection of all query results"""
    listQueryResults: List[OneQueryResultList] = Field(default=None, description="List of OneQueryResultList - all result")
    rrfScores : RRFScores = Field(default=None, description="RRF ranks in descending order")
