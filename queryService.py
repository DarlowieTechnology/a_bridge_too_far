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

from common import COLLECTION, TOKENIZERTYPES, ConfigCollection, MatchingChunks, AllTopicMatches, ChunkInfo, OpenFile
from resultsQueryClasses import SEARCH, OneQueryAppResult, OneQueryResultList, RRFScores, IdentifierQueryResults, AllQueryResults




class QueryService(BaseModel):


    def semanticQuery(self, query : List[str], chromaCollection : Collection, queryLabel : str, maxRetrieveNumber : int, maxCutItemDistance : int) -> OneQueryResultList:
        """
        Performs semantic query. Returns list of results
        Use maxCutItemDistance value to cut results off
        Use maxRetrieveNumber to limit max number of items returned
        
        :param query: query for semantic search
        :type query: List[str]
        :param chromaCollection: chroma DB collection
        :type chromaCollection: Collection
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :param maxRetrieveNumber: maximum number of results to return
        :type maxRetrieveNumber: int
        :param maxCutItemDistance: maximum distance of the result
        :type maxCutItemDistance: int
        :return: list of results
        :rtype: OneQueryResultList
        """

        oneQueryResultList = OneQueryResultList(
            query ={'searchType' : SEARCH.SEMANTIC, 'query' : query },
            label = queryLabel        
        )

        queryResult = chromaCollection.query(query_texts = query, n_results = maxRetrieveNumber)

        resultIdx = -1

        for distFloat in queryResult["distances"][0]:
            resultIdx += 1
            if (distFloat > maxCutItemDistance) :
                break

#            self.printOneQueryResult(queryResult, resultIdx)

            oneQueryResultList.appendQueryChunkResult(
                score = distFloat,
                rank = resultIdx,
                chunk = queryResult["documents"][0][resultIdx],
                chunkID = queryResult["metadatas"][0][resultIdx]["chunkid"],
                document = queryResult["metadatas"][0][resultIdx]["document"],
                searchTypeName = SEARCH.SEMANTIC.value
            )
        return oneQueryResultList


    def bm25sQuery(self, query : List[List[str]], folderName : str, queryLabel : str, bm25sRetrieveNumber : int, bm25sMinCutOffScore : float) -> OneQueryResultList : 
        """
        Perform bm25s query for combined corpus of documents
        data in corpus is encoded as documentFileName + '--' + chunkId + '\n' + chunkText
        Number of items retrieved is limited to min of context['bm25sRetrieveNum'] and number of items
        Discard items with score less or equal to value context['bm25sCutOffScore']

        :param query: list of lists of query tokens for bm25s
        :type query: List[List[str]]
        :param folderName: name of folder with bm25s index
        :type folderName: str
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :param bm25sRetrieveNumber: number of results to retrieve
        :type bm25sRetrieveNumber: int
        :param bm25sMinCutOffScore: cut off for bm25s score, items with lower score are discarded
        :type bm25sMinCutOffScore: float
        :return: search result object
        :rtype: OneQueryResultList
        """

        oneQueryResultList = OneQueryResultList(
            query ={'searchType' : SEARCH.BM25S, 'query' : query },
            label = queryLabel        
        )

        retriever = bm25s.BM25.load(save_dir=str(folderName), mmap=True, load_corpus=True)

        max_items = bm25sRetrieveNumber
        if retriever.scores["num_docs"] < bm25sRetrieveNumber:
            max_items = retriever.scores["num_docs"]

        results, scores = retriever.retrieve(query, k=max_items)
        for rankIdx in range(results.shape[1]):
            docN, score = results[0, rankIdx], scores[0, rankIdx]
            if bm25sMinCutOffScore >= score:
                break
            docN = docN["text"].splitlines()
            combo = docN[0].strip()
            documentAndID = combo.split('--')
            documentName = documentAndID[0]
            chunkID = documentAndID[1]
            chunkText = docN[1].strip()
            oneQueryResultList.appendQueryChunkResult(
                score = score,
                rank = rankIdx,
                chunk = chunkText,
                chunkID = chunkID,
                document = documentName,
                searchTypeName = SEARCH.BM25S.value
            )
        return oneQueryResultList


    def rrfReRanking(self, allQueryResults : AllQueryResults) -> AllQueryResults:
        """
        Reciprocal Rank Fusion (RRF) re-ranking of semantic and bm25s search results.
        RRF rank is used as dict key. It is rounded to 6 digits
        
        :param allQueryResults: query results
        :type allQueryResults: AllQueryResults
        :return: query results updated with rank
        :rtype: AllQueryResults
        """

        # merge identifiers in the form "document names--chunkID" from all runs into set
        setKeys = set()
        for item in allQueryResults.listQueryResults:
            for key in item.result_dict:
                setKeys.add(key)

        # print(f"RRF: semantic: {len(allQueryResults.listQueryResults[0].result_dict)}  bm25s: {len(allQueryResults.listQueryResults[1].result_dict)}  unique keys : {len(setKeys)}")

        scoresDict = dict[str, IdentifierQueryResults]()

        # calculate rank for issue access all query runs
        for ident in list(setKeys):
            identifierQueryResults = IdentifierQueryResults(
                identifier = ident,
            )
            finalScore = 0.0
            for item in allQueryResults.listQueryResults:               # item : OneQueryResultList
                if ident in item.result_dict.keys():
                    oneQueryChunkResult = item.result_dict[ident]        # OneQueryChunkResult
                    finalScore += 1/(60 + oneQueryChunkResult.score)
                    if oneQueryChunkResult.searchTypeName == SEARCH.SEMANTIC.value:
                        identifierQueryResults.chunk = oneQueryChunkResult.chunk
            identifierQueryResults.score = finalScore
            scoresDict[ident] = identifierQueryResults

        # sort descending by rank
        scoresDict = {k: v for k, v in sorted(scoresDict.items(), key=lambda item: item[1].score, reverse=True)}

        allQueryResults.rrfScores = RRFScores(
            scoresDict = scoresDict
        )

        # print(f"=====\n{allQueryResults.rrfScores.model_dump_json(indent=2)}\n================")

        return allQueryResults



    def printOneQueryResult(self, queryResult, resultIdx) :
        print(f"----{resultIdx}------------")
        dist = queryResult["distances"][0][resultIdx]
#        docType = type(queryResult["documents"][0][resultIdx])
        doc = queryResult["documents"][0][resultIdx]
#        metaType = type(queryResult["metadatas"][0][resultIdx])
        meta = queryResult["metadatas"][0][resultIdx]
        print(f"distance: {dist}")
#        print(f"doc type: {docType}")
        print(f"doc: {doc}")
#        print(f"metadata type:{metaType}")
        print(f"metadata: {meta}")


    def hydeQuery(self, query : List[str], model : Model) -> tuple[str, RunUsage]:
        """ Use HyDE (Hypothetical Document Embedding) to improve the query for semantic search. Throws exceptions on LLM errors.

            :param query: query for semantic search
            :type query: List[str]
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
        return result.output, result.usage()


    def multiQuery(self, query : List[str], model : Model) -> tuple[List[str], RunUsage]:
        """Generate multiple queries form the original query for semantic search. Throws exceptions on LLM errors.
        
        :param query: query for semantic search
        :type query: List[str]
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
            output_type = List,
            retries = 3)
        userPrompt = query
        result = agentMultipleQ.run_sync(userPrompt)
        return result.output, result.usage()


    def rewriteQuery(self, query : List[str], model : Model) -> tuple[str, RunUsage]:
        """
        Rewrite the query for semantic search.  Throws exceptions on LLM errors.
        """
        # Query rewriting prompt
        systemPrompt = """You are a query rewriting expert. The user's original query didn't retrieve relevant documents.

            Analyze the query and rewrite it to improve retrieval chances:
            - Make it more specific or more general as appropriate
            - Add synonyms or related terms
            - Rephrase to target likely document content
            - Consider the retrieval failure and adjust accordingly.
            Output only the query, do not include additional information.

            Original query is supplied in user prompt.
            """

        agentRewriteQ = Agent(
            model = model, 
            system_prompt = systemPrompt,
            retries = 3)
        userPrompt = query
        result = agentRewriteQ.run_sync(userPrompt)
        return result.output, result.usage()


    def tokenizeQuery(self, query : List[str], tokenizerTypes: TOKENIZERTYPES) -> List[str]:
        """
        create list of tokens from the query for BM25S search.

        :param query: query for semantic search
        :type query: List[str]
        :param tokenizerTypes: flags for tokenizer
        :type tokenizerTypes: TOKENIZERTYPES
        :return: List of token strings
        :rtype: List[str]
        """

        if TOKENIZERTYPES.STOPWORDSEN in tokenizerTypes:
            stopwords = "english"
        else:
            stopwords = None

        if TOKENIZERTYPES.STEMMER in tokenizerTypes:
            stemmer=Stemmer.Stemmer("english")
        else:
            stemmer = None

        query_tokens = bm25s.tokenize(texts = query, return_ids=False, stopwords=stopwords, stemmer=stemmer)
        return query_tokens
