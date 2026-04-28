from typing import List, Dict
from typing_extensions import Self
import json

import Stemmer
import bm25s


from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from openai import Model

import chromadb
from chromadb import Collection

import numpy as np

from common import COLLECTION, TOKENIZERTYPES, ConfigCollection, MatchingChunks, AllTopicMatches, ChunkInfo, OpenFile
from resultsQueryClasses import SEARCH, RRFScores, IdentifierQueryResults, OneQueryChunkResult, OneChunkQueryResultList, AllChunkQueryResults



class StatsOnList(BaseModel) :
    """represents statistics of data set"""
    length : float = Field(default=0, description="length of dataset")
    min : float = Field(default=0, description="minimum value in the dataset") 
    max : float = Field(default=0, description="maximum value in the dataset") 
    avg : float = Field(default=0, description="average value in the dataset") 
    mean : float = Field(default=0, description="mean value in the dataset") 
    median : float = Field(default=0, description="median value in the dataset") 
    range : float = Field(default=0, description="range of values in the dataset") 
    q1 : float = Field(default=0, description="1st quartile of the dataset") 
    q2 : float = Field(default=0, description="2nd quartile of the dataset") 
    q3 : float = Field(default=0, description="3nd quartile of the dataset") 


    def makeStatsOnList(self, scoresForStats) :
        """
        Calculate stats on the list of float
        
        :param scoresForStats: list of float
        """

        if not len(scoresForStats):
            return

        a1F = np.array(scoresForStats, dtype=np.float32)

        self.length = len(a1F)
        self.min = np.min(a1F)
        self.max = np.max(a1F)
        self.avg = np.average(a1F)
        self.mean = np.mean(a1F)
        self.median = np.median(a1F)
        self.range = np.max(a1F)-np.min(a1F)
        self.q1, self.q2, self.q3 = np.quantile(a1F, [0.25, 0.5, 0.75])

    def find_outliers_zscore(self, scoresForStats, threshold=3):
        mean_val = np.mean(scoresForStats)
        std_dev = np.std(scoresForStats)
        z_scores = [(y - mean_val) / std_dev for y in scoresForStats]
        # Identify outliers using absolute z-score
        outliers = [y for y, z_score in zip(scoresForStats, z_scores) if np.abs(z_score) > threshold]
        # Filter non-outlier data (optional)
        # clean_data = data[np.abs(z_scores) <= threshold] 
        return outliers



class QueryService(BaseModel):


    def semanticQuery(self, query : str, chromaCollection : Collection, queryLabel : str, maxRetrieveNumber : int, maxCutItemDistance : int) -> OneChunkQueryResultList:
        """
        Performs semantic query. Returns list of results
        Use maxCutItemDistance value to cut results off
        Use maxRetrieveNumber to limit max number of items returned
        
        :param query: query for semantic search
        :type query: str
        :param chromaCollection: chroma DB collection
        :type chromaCollection: Collection
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :param maxRetrieveNumber: maximum number of results to return
        :type maxRetrieveNumber: int
        :param maxCutItemDistance: maximum distance of the result
        :type maxCutItemDistance: int
        :return: list of results
        :rtype: OneChunkQueryResultList
        """

        oneChunkQueryResultList = OneChunkQueryResultList(
            query = query,
            label = queryLabel
        )

        queryResult = chromaCollection.query(query_texts = query, n_results = maxRetrieveNumber)

#        print(f"SEMANTIC=====\n'{queryLabel}' : {query}\n===================")

        resultIdx = -1
        for distFloat in queryResult["distances"][0]:
            resultIdx += 1                              # index starts from 0
            if (distFloat > maxCutItemDistance) :
                break

            oneQueryChunkResult = OneQueryChunkResult(
                score = distFloat,
                rank = resultIdx + 1,                   # rank starts from 1
                chunk = queryResult["documents"][0][resultIdx],
                chunkID = queryResult["metadatas"][0][resultIdx]["chunkid"],
                document = queryResult["metadatas"][0][resultIdx]["document"],
                searchTypeName = SEARCH.SEMANTIC.value
            )
            ident = queryResult["metadatas"][0][resultIdx]["document"] + "|" + queryResult["metadatas"][0][resultIdx]["chunkid"]
            oneChunkQueryResultList.appendQueryResult(
                identifier = ident,
                queryResult = oneQueryChunkResult
            )

#            print(f"Rank:{resultIdx+1}   score: {distFloat}    doc: {queryResult["metadatas"][0][resultIdx]["document"]}   chunk ID: {queryResult["metadatas"][0][resultIdx]["chunkid"]}")
#            print(f"chunk:\n{queryResult["documents"][0][resultIdx]}")

        return oneChunkQueryResultList


    def bm25sQuery(self, query : List[str], folderName : str, queryLabel : str, bm25sRetrieveNumber : int, bm25sMinCutOffScore : float) -> OneChunkQueryResultList : 
        """
        Perform bm25s query for combined corpus of documents
        data in corpus is encoded as documentFileName + '|' + chunkId + '\n' + chunkText
        Number of items retrieved is limited to min of context['bm25sRetrieveNum'] and number of items
        Discard items with score less or equal to value context['bm25sCutOffScore']

        :param query: lists of query tokens for bm25s
        :type query: List[str]
        :param folderName: name of folder with bm25s index
        :type folderName: str
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :param bm25sRetrieveNumber: number of results to retrieve
        :type bm25sRetrieveNumber: int
        :param bm25sMinCutOffScore: cut off for bm25s score, items with lower score are discarded
        :type bm25sMinCutOffScore: float
        :return: search result object
        :rtype: OneChunkQueryResultList
        """

        combinedQuery = " ".join(query)
        oneChunkQueryResultList = OneChunkQueryResultList(
            query = combinedQuery,
            label = queryLabel        
        )

#        print(f"BM25S=========\n'{queryLabel}' : {query}\n===================")

        retriever = bm25s.BM25.load(save_dir=str(folderName), mmap=True, load_corpus=True)

        max_items = bm25sRetrieveNumber
        if retriever.scores["num_docs"] < bm25sRetrieveNumber:
            max_items = retriever.scores["num_docs"]

        results, scores = retriever.retrieve([query], k=max_items)
        for rankIdx in range(results.shape[1]):
            docN, score = results[0, rankIdx], scores[0, rankIdx]
            if bm25sMinCutOffScore >= score:
                break
            docN = docN["text"].splitlines()
            ident = docN[0].strip()
            documentAndID = ident.split('|')
            documentName = documentAndID[0]
            chunkID = documentAndID[1]
            chunkText = docN[1].strip()

            oneQueryChunkResult = OneQueryChunkResult(
                score = score,
                rank = rankIdx + 1,                   # rank starts from 1
                chunk = chunkText,
                chunkID = chunkID,
                document = documentName,
                searchTypeName = SEARCH.BM25S.value
            )
            oneChunkQueryResultList.appendQueryResult(
                identifier = ident,
                queryResult = oneQueryChunkResult
            )

#            print(f"Rank:{rankIdx+1}  score: {score}    doc: {documentName}   chunk ID: {chunkID}")
#            print(f"chunk:\n{chunkText}")

        return oneChunkQueryResultList


    def rrfReRanking(self, allChunkQueryResults : AllChunkQueryResults) -> AllChunkQueryResults:
        """
        Reciprocal Rank Fusion (RRF) re-ranking of semantic and bm25s search results.
        
        :param allChunkQueryResults: query results
        :type allChunkQueryResults: AllChunkQueryResults
        :return: query results updated with rank
        :rtype: AllChunkQueryResults
        """

        # merge identifiers in the form "document name|chunkID" from all runs into set
        setKeys = set()
        for item in allChunkQueryResults.listQueryResults:
            for key in item.result_dict:
                setKeys.add(key)

        scoresDict = dict[str, IdentifierQueryResults]()

        # calculate rank for issue access all query runs
        for ident in list(setKeys):
            identifierQueryResults = IdentifierQueryResults(
                identifier = ident,
            )
            finalRank = 0.0
            for item in allChunkQueryResults.listQueryResults:               # item : OneQueryResultList
                if ident in item.result_dict.keys():
                    oneQueryChunkResult = item.result_dict[ident]        # OneQueryChunkResult
                    finalRank += 1/(60 + oneQueryChunkResult.rank)
            identifierQueryResults.rrfRank = finalRank
            scoresDict[ident] = identifierQueryResults

        # sort descending by rank
        scoresDict = {k: v for k, v in sorted(scoresDict.items(), key=lambda item: item[1].rrfRank, reverse=True)}

        position = 0
        # initialize position field
        for key in scoresDict.keys():
            identifierQueryResults = scoresDict[key]
            identifierQueryResults.position = position
            position += 1

        allChunkQueryResults.rrfScores = RRFScores(
            scoresDict = scoresDict
        )

#        print(f"=====\n{allQueryResults.rrfScores.model_dump_json(indent=2)}\n================")

        return allChunkQueryResults


    def hydeQuery(self, query : str, model : Model) -> tuple[str, RunUsage]:
        """ Use HyDE (Hypothetical Document Embedding) to improve the query for semantic search. Throws exceptions on LLM errors.

            :param query: query for semantic search
            :type query: str
            :param model: OpenAI model instance
            :type model: Model
            :return: tuple of results and run usage
            :rtype: tuple[str, RunUsage]

        """

        systemPrompt = f"Write a two sentence answer to the user prompt query"

        agentHyDE = Agent(
            model = model, 
            system_prompt = systemPrompt,
            retries = 3)
        userPrompt = query
        result = agentHyDE.run_sync(userPrompt)
        resOutput: str = ""
        if type(result.output) == str:
            resOutput = result.output
        if type(result.output) == list:
            resOutput = " ".join(result.output)
        return resOutput, result.usage()


    def multiQuery(self, query : str, model : Model) -> tuple[str, RunUsage]:
        """Generate multiple queries form the original query for semantic search. Throws exceptions on LLM errors.
        
        :param query: query for semantic search
        :type query: str
        :param model: OpenAI model instance
        :type model: Model
        :return: tuple of results and run usage
        :rtype: tuple[str, RunUsage]
        """

        # Prompt for generating multiple queries
        systemPrompt = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Respond only with a list of questions, do not include additional information. Format output as a list of strings.
        Original query is supplied in user prompt"""

        agentMultipleQ = Agent(
            model = model, 
            system_prompt = systemPrompt, 
            output_type = str,
            retries = 3)
        userPrompt = query
        result = agentMultipleQ.run_sync(userPrompt)
        resOutput: str = ""
        if type(result.output) == str:
            resOutput = result.output
        if type(result.output) == list:
            resOutput = " ".join(result.output)
        return resOutput, result.usage()


    def rewriteQuery(self, query : str, model : Model) -> tuple[str, RunUsage]:
        """
        Rewrite the query for semantic search.  Throws exceptions on LLM errors.
        
        :param query: query for semantic search
        :type query: str
        :param model: OpenAI model instance
        :type model: Model
        :return: tuple of results and run usage
        :rtype: tuple[str, RunUsage]
        """
        # Query rewriting prompt
        systemPrompt = """You are a query rewriting expert. The user's original query didn't retrieve relevant documents.

            Analyze the query and rewrite it to improve retrieval chances:
            - Make it more specific or more general as appropriate
            - Add synonyms or related terms
            - Rephrase to target likely document content
            - Consider the retrieval failure and adjust accordingly.
            Output only the query, do not include additional information. Format output as a list of strings.

            Original query is supplied in user prompt.
            """

        agentRewriteQ = Agent(
            model = model, 
            system_prompt = systemPrompt,
            retries = 3)
        userPrompt = query
        result = agentRewriteQ.run_sync(userPrompt)
        resOutput: str = ""
        if type(result.output) == str:
            resOutput = result.output
        if type(result.output) == list:
            resOutput = " ".join(result.output)
        return resOutput, result.usage()


    def tokenizeQuery(self, query : str) -> List[str]:
        """
        create list of tokens from the query for BM25S search.

        :param query: query for semantic search
        :type query: str
        :return: List of token strings
        :rtype: List[str]
        """

        query_tokens = bm25s.tokenize(texts = query, return_ids=False, stopwords="english")
        return query_tokens[0]


    def getOutliersForQuery(self, oneChunkQueryResultList : OneChunkQueryResultList, threshold : float = 3.0, upper : bool = True) -> List[str]:
        """
        Make list of scores from oneQueryResultList. Round scores to 10.
        Sort the list in ascending order. Calculate IQR and Z scores
        Return list of identifiers for outlier chunks.

        :param oneChunkQueryResultList: one query results
        :type oneChunkQueryResultList: OneChunkQueryResultList
        :param threshold: Z Score threshold - how many standard deviations away the value is
        :type threshold: float
        :param upper: IQR upper or lower fence
        :type upper: bool
        :return: list of of outlier identifiers
        :rtype: List[str]
        """

        inData: List[float] = []
        for ident in oneChunkQueryResultList.result_dict.keys():
            oneQueryChunkResult = oneChunkQueryResultList.result_dict[ident]
            inData.append(round(oneQueryChunkResult.score, 10))

        inData = sorted(inData)

        data = np.array(inData)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upperFence = q3 + (1.5 * iqr)
        lowerFence = q1 - (1.5 * iqr)

        mean_val = np.mean(inData)
        std_dev = np.std(inData)
        z_scores = [(y - mean_val) / std_dev for y in inData]
        outliers = [y for y, z_score in zip(inData, z_scores) if z_score > threshold]

        outList : List[str] = []
        for ident in oneChunkQueryResultList.result_dict.keys():
            oneQueryChunkResult = oneChunkQueryResultList.result_dict[ident]
            key = round(oneQueryChunkResult.score, 10)
            if key in outliers:
                outList.append(ident)

        outlierScores: List[float] = []
        for val in inData:
            if upper:
                if val > upperFence:
                    outlierScores.append(val)
            else:
                if val < lowerFence:
                    outlierScores.append(val)
        
        for val in outlierScores:
            for ident in oneChunkQueryResultList.result_dict.keys():
                oneQueryChunkResult = oneChunkQueryResultList.result_dict[ident]
                if round(oneQueryChunkResult.score, 10) == val:
                    outList.append(ident)    
                    break
        return outList


    def getOutliersFromRRF(self, allChunkQueryResults : AllChunkQueryResults, iqrCoefficient : float, zScoreThreshold : float) -> Dict[str, IdentifierQueryResults]:
        """
        Select outliers in array of RRF ranks by Interquartile Range (IQR) and Z Score

        :param allChunkQueryResults: all query results
        :type allChunkQueryResults: AllChunkQueryResults
        :param iqrCoefficient: IQR  coefficient - scaling upper fence in skewed distributions
        :type iqrCoefficient: float
        :param zScoreThreshold: Z Score threshold - how many standard deviations away the value is
        :type zScoreThreshold: float
        :return: dict of outliers, key is ident, value is IdentifierQueryResults object
        :rtype: Dict[str, IdentifierQueryResults]
        """

        inData: List[float] = []
        for ident in allChunkQueryResults.rrfScores.scoresDict.keys():
            identifierQueryResults = allChunkQueryResults.rrfScores.scoresDict[ident]
            inData.append(round(identifierQueryResults.rrfRank, 10))
        
        inData = sorted(inData)

        data = np.array(inData)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upperFence = q3 + (iqrCoefficient * iqr)

        mean_val = np.mean(inData)
        std_dev = np.std(inData)
        # if standard deviation is zero - make non-zero
        if std_dev == 0:
            z_scores = [(y - mean_val) / 0.0000001 for y in inData]
        else:
            z_scores = [(y - mean_val) / std_dev for y in inData]
        outliers = [y for y, z_score in zip(inData, z_scores) if z_score > zScoreThreshold]

        outDict : Dict[str, IdentifierQueryResults] = {}
        for ident in allChunkQueryResults.rrfScores.scoresDict.keys():
            identifierQueryResults = allChunkQueryResults.rrfScores.scoresDict[ident]
            val = round(identifierQueryResults.rrfRank, 10)
            if val > upperFence:
                identifierQueryResults.outlierIQR = True
            if val in outliers:
                identifierQueryResults.outlierZScore = True
            if identifierQueryResults.outlierIQR or identifierQueryResults.outlierZScore:
                outDict[ident] = identifierQueryResults

        return outDict
