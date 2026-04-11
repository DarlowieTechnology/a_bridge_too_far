#
# Query workflow class used by Django app and command line
#
import sys
import json
from logging import Logger
from typing import List, Dict
from typing_extensions import Self
from pathlib import Path
import time
import re


import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic import BaseModel, Field, model_validator
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage


from jira import JIRA
from openai import OpenAI

import Stemmer
import bm25s
import spacy
from anyascii import anyascii


# local
from common import COLLECTION, QUERYTYPES, TOKENIZERTYPES, OneResultWithType, ResultWithTypeList, ConfigCollection
from resultsQueryClasses import SEARCH, OneIndexerQueryResult, IdentifierQueryResults, RRFScores, OneIndexerQueryResultList, AllIndexerQueryResults
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory


class QueryWorkflow(WorkflowBase):

    dataFolder : str = Field(default = "", description="Intermediate data folder")
    query : str = Field(default = "", description="List of queries for this workflow")
    queryTransforms : QUERYTYPES = Field(default = "", description="List of query transformation flags")
    bm25IndexFolder : str = Field(default = "", description="bm25 index folder")
    bm25CorpusFileName : str = Field(default = "", description="bm25 corpus file")
    semanticMaxCutItemDistance : float = Field(default = 0.5, description = "Maximum distance in semantic search")
    semanticRetrieveNumber : int = Field(default = 512, description = "Number of items retrieved with semantic query")
    queryBM25Options : TOKENIZERTYPES = Field(default = TOKENIZERTYPES.STOPWORDSEN, description = "Options for BM25 query tokenizer")
    bm25sMinCutOffScore : float = Field(default = 0.0, description="Minimum bm25s score cut off")
    bm25sRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with bm25s query")
    outputNumber : int = Field(default = 1, description="Maximum number of items to return")
    outputFileName : str = Field(default = "", description="File name for results")
    queryPreprocessFlag : bool = Field(default = True, description="Call preprocessQuery() after every query transform")
    queryCompressFlag : bool = Field(default = True, description="Call Telegraphic Semantic Compression (TSC) after every query transform")

    queryHyDE : str = Field(default = "", description="HyDE query value")
    queryMultiple : str = Field(default = "", description="Multiple query value")
    queryRewrite : str = Field(default = "", description="Rewrite query value")
    querybm25sprep : str = Field(default = "", description="bm25s query value")
    queryTokenized : str = Field(default = "", description="bm25s tokenized query value")

    stats : Dict[str, int] = Field(default = {}, description="Run statistics")


#    @model_validator(mode='after')
    def queryWorkflow_verify_configuration(self) -> Self:
        if not Path(self.dataFolder).is_dir:
            raise ValueError(f'Intermediate data folder is invalid')
        if not Path(self.dataFolder + self.bm25IndexFolder).is_dir:
            raise ValueError(f'bm25 index folder is invalid')
        if not self.semanticRetrieveNumber in range(0, 2049):
            raise ValueError(f'Number of semantic search items is invalid')
        if not (self.semanticMaxCutItemDistance >= 0 and self.semanticMaxCutItemDistance <= 1.0):
            raise ValueError(f'Maximum distance of semantic search items is invalid')
        if not (self.queryBM25Options >= 0 and self.queryBM25Options <= 1.0):
            raise ValueError(f'Minimum bm25s score cut off is invalid')
        if not self.bm25sRetrieveNumber in range(0, 2049):
            raise ValueError(f'Number of bm25s search items is invalid')
        if not self.outputNumber in range(1, 100):
            raise ValueError(f'output number is invalid')
        if not Path(self.outputFileName).is_file:
            raise ValueError(f'Output file name is invalid')

        return Self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        self.dataFolder = configCollection["dataFolder"]

        if configCollection.keyExists("query"): 
            self.query = configCollection["query"]
        if configCollection.keyExists("queryTransforms"): 
            self.queryTransforms = configCollection["queryTransforms"]
        if configCollection.keyExists("bm25IndexFolder"): 
            self.bm25IndexFolder = configCollection["bm25IndexFolder"]
        if configCollection.keyExists("bm25CorpusFileName"):
            self.bm25CorpusFileName = configCollection["bm25CorpusFileName"]
        if configCollection.keyExists("semanticMaxCutItemDistance"):
            self.semanticMaxCutItemDistance = configCollection["semanticMaxCutItemDistance"]
        if configCollection.keyExists("semanticRetrieveNumber"):
            self.semanticRetrieveNumber = configCollection["semanticRetrieveNumber"]
        if configCollection.keyExists("queryBM25Options"):
            self.queryBM25Options = configCollection["queryBM25Options"]
        if configCollection.keyExists("bm25sMinCutOffScore"):
            self.bm25sMinCutOffScore = configCollection["bm25sMinCutOffScore"]
        if configCollection.keyExists("bm25sRetrieveNumber"):
            self.bm25sRetrieveNumber = configCollection["bm25sRetrieveNumber"]
        if configCollection.keyExists("outputNumber"):
            self.outputNumber = configCollection["outputNumber"]
        if configCollection.keyExists("outputFileName"):
            self.outputFileName = configCollection["outputFileName"]
        if configCollection.keyExists("queryPreprocess"):
            self.queryPreprocessFlag = configCollection["queryPreprocess"]
        if configCollection.keyExists("queryCompress"):
            self.queryCompressFlag = configCollection["queryCompress"]

        self.stats = {}

        # manually call model validator
        self.queryWorkflow_verify_configuration()


    def preprocessQuery(self, queryStr : str) :
        """
        Preprocess string for interaction with LLM: convert all characters to ASCII, lowercase, remove whitespace, normalise spaces
        Can be used on user input or output of LLM

        :param query: original query 
        :type query: str
        :return: preprocessed query
        :rtype: str
        """
        query = anyascii(queryStr)
        query = query.strip().lower()
        query = re.sub(r'[^\w\s?!]', '', query)
        query = " ".join(query.split())
        return query


    def compressQuery(self, query : str) -> str:
        """
        Perform Telegraphic Semantic Compression (TSC) on the query for semantic search. 
        Ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/.
        Get english dictionary: python -m spacy download en_core_web_sm.

        :param query:  original query 
        :type query: str
        :return: compressed query
        :rtype: str
        """

        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Parts of speech to remove (predictable grammar)
        REMOVE_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}

        # Optional low-information words to remove
        REMOVE_LIKE = {"like", "just", "really", "basically", "literally"}

        doc = nlp(query)
        chunks = []

        for sent in doc.sents:
            words = [
                token.lemma_
                for token in sent
                if (
                    token.pos_ not in REMOVE_POS
                    and token.text.lower() not in REMOVE_LIKE
                    and not token.is_punct
                )
            ]
            if words:
                chunks.append(" ".join(words))
        newQuery =  " ".join(chunks)
        return newQuery

    
    def tokenizeQuery(self, query : str, tokenizerTypes: TOKENIZERTYPES) -> str:
        """
        create list of tokens from the query for BM25S search.
        Overwrite self.queryTokenized with new value.
        
        :param useStopWordsFlag:  remove stopwords from the query.
        :type useStopWordsFlag: bool
        :param useStemmerFlag: reduce words in query to stems .
        :type useStemmerFlag: bool
        """

        if TOKENIZERTYPES.STOPWORDSEN in tokenizerTypes:
            stopwords = "english"
        else:
            stopwords = None

        if TOKENIZERTYPES.STEMMER in tokenizerTypes:
            stemmer=Stemmer.Stemmer("english")
        else:
            stemmer = None

        query_tokens = bm25s.tokenize(query, return_ids=False, stopwords=stopwords, stemmer=stemmer)
        self.queryTokenized = query_tokens
        return query_tokens


    def hydeQuery(self, query : str) -> str:
        """ Use HyDE (Hypothetical Document Embedding) to improve the query for semantic search. 
            Overwrite self.queryHyDE with new value.
        """

        startHyDE = time.time()
        systemPrompt = f"Write a two sentence answer to the user prompt query"

        agentHyDE = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentHyDE.run_sync(userPrompt)
            self.queryHyDE = result.output
            self.addUsage(result.usage())
            endHyDE = time.time()
            if result.usage():
                msg = f"Query HyDE: Usage: {self.usageFormat(usage = result.usage(), insertHTML = False)}. Time: {(endHyDE - startHyDE):9.2f} seconds."
            else:
                msg = f"Query HyDE: Time: {(endHyDE - startHyDE):9.2f} seconds."
            self.workerError(msg)
            return self.queryHyDE
        except Exception as e:
            msg = f"LLM exception on HyDE request: {e}"
            self.workerError(msg)
        return ""


    def multiQuery(self, query : str) -> str:
        """Generate multiple queries form the original query for semantic search. 
            Overwrite self.queryMultiple with new value.
        """

        # Prompt for generating multiple queries
        systemPrompt = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Respond only with a list of questions. Format output as Python list.
        Original query is supplied in user prompt"""

        startMulti = time.time()
        agentMultipleQ = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentMultipleQ.run_sync(userPrompt)
            self.queryMultiple = result.output
            self.addUsage(result.usage())
            endMulti = time.time()
            if result.usage():
                msg = f"Query Multi: Usage: {self.usageFormat(usage = result.usage(), insertHTML = False)}. Time: {(endMulti - startMulti):9.2f} seconds."
            else:
                msg = f"Query Multi: Time: {(endMulti - startMulti):9.2f} seconds."
            self.workerError(msg)
            return self.queryMultiple
        except Exception as e:
            msg = f"LLM exception on multi query request: {e}"
            self.workerError(msg)


    def rewriteQuery(self, query : str) -> str:
        """
        Rewrite the query for semantic search. 
        Overwrite self.queryRewrite with new value.
        """
        # Query rewriting prompt
        systemPrompt = """You are a query rewriting expert. The user's original query didn't retrieve relevant documents.

            Analyze the query and rewrite it to improve retrieval chances:
            - Make it more specific or more general as appropriate
            - Add synonyms or related terms
            - Rephrase to target likely document content
            - Consider the retrieval failure and adjust accordingly

            Original query is supplied in user prompt.
            """
        
        startRewrite = time.time()
        agentRewriteQ = Agent(self._llmModel, 
                               system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentRewriteQ.run_sync(userPrompt)
            self.queryRewrite = result.output
            self.addUsage(result.usage())
            endRewrite = time.time()
            if result.usage():
                msg = f"Query Rewrite: Usage: {self.usageFormat(usage = result.usage(), insertHTML = False)}. Time: {(endRewrite - startRewrite):9.2f} seconds."
            else:
                msg = f"Query Rewrite: Time: {(endRewrite - startRewrite):9.2f} seconds."
            return self.queryRewrite
        except Exception as e:
            msg = f"LLM exception on rewrite query request: {e}"
            self.workerError(msg)
        return ""


    def prepBM25S(self, query : str) -> str:
        """Prepare query for BM25S search using LLM. 
        Overwrite self.querybm25sprep with new value.
        """

        # Prompt for generating prepared query
        systemPrompt = """You are a cybersecurity expert. 
        Return a list of terms in the original user prompt with their descriptions.
        Return only the results. Format output as a list of Python strings.
        Original query is supplied in user prompt"""

        startPrepBM25s = time.time()
        agentPrepBM25s = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentPrepBM25s.run_sync(userPrompt)
            self.addUsage(result.usage())
            self.querybm25sprep = result.output
            endPrepBM25s = time.time()
            if result.usage():
                msg = f"Query BM25s Prep: Usage: {self.usageFormat(usage = result.usage(), insertHTML = False)}. Time: {(endPrepBM25s - startPrepBM25s):9.2f} seconds."
            else:
                msg = f"Query BM25s Prep: Time: {(endPrepBM25s - startPrepBM25s):9.2f} seconds."
            return self.querybm25sprep
        except Exception as e:
            msg = f"LLM exception on prepBM25S query request: {e}"
            self.workerError(msg)



    def bm25sQuery(self, query : str, folderName : str, queryLabel : str) -> OneIndexerQueryResultList : 
        """
        Perform bm25s query for combined corpus of documents
        data in corpus is encoded as 'identifier\\ntitle'
        Number of items retrieved is limited to min of self.bm25sRetrieveNumber and number of items
        Discard items with score less or equal to value self.bm25sMinCutOffScore

        :param query: query for bm25s
        :type query: str
        :param folderName: name of folder with bm25s index
        :type folderName: str
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :return: search result object
        :rtype: OneIndexerQueryResultList
        """

        oneIndexerQueryResultList = OneIndexerQueryResultList(
            query ={'searchType' : SEARCH.BM25S, 'query' : query },
            label = queryLabel
        )

        retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)

        max_items = self.bm25sRetrieveNumber
        if retriever.scores["num_docs"] < self.bm25sRetrieveNumber:
            max_items = retriever.scores["num_docs"]

        results, scores = retriever.retrieve(query, k=max_items)
        for rankIdx in range(results.shape[1]):
            docN, score = results[0, rankIdx], scores[0, rankIdx]
            docN = docN["text"].splitlines()
            if (score > self.bm25sMinCutOffScore):
                oneIndexerQueryResult = OneIndexerQueryResult(
                    score = score,
                    rank = rankIdx + 1,             # rank starts from 1
                    identifier = docN[0].strip(),
                    title = docN[1].strip(),
                    report = str(Path(folderName).stem)
                )
                oneIndexerQueryResultList.appendQueryResult(
                    identifier = docN[0].strip(),
                    queryResult = oneIndexerQueryResult
                )
        return oneIndexerQueryResultList


    def vectorQuery(self, query : str, collection : COLLECTION, queryLabel : str) -> OneIndexerQueryResultList:
        """
        Performs vector (semantic) query. Returns list of results
        Uses semanticMaxCutItemDistance to cut results off
        Uses semanticRetrieveNumber to limit max number of items
        
        :param query: query for semantic search
        :type query: str
        :param collection: chroma DB collection name for query
        :type query: COLLECTION
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :return: list of results
        :rtype: OneIndexerQueryResultList
        """

        oneIndexerQueryResultList = OneIndexerQueryResultList(
            query ={'searchType' : SEARCH.SEMANTIC, 'query' : [ query ] },
            label = queryLabel
        )

        chromaCollection = self.collections[collection]
        queryResult = chromaCollection.query(query_texts=query, n_results=self.semanticRetrieveNumber)

        resultIdx = -1

        for distFloat in queryResult["distances"][0]:
            resultIdx += 1
            if (distFloat > self.semanticMaxCutItemDistance) :
                break

    #            print(f"------dist {distFloat}-------------------")
    #            print(type(queryResult["documents"][0][resultIdx]))
    #            print(queryResult["documents"][0][resultIdx])
    #            print("-------------------------")
    #            print(type(queryResult["metadatas"][0][resultIdx]))
    #            print(queryResult["metadatas"][0][resultIdx])

            IssueTemplate = ParserClassFactory.factory(queryResult["metadatas"][0][resultIdx]["recordType"])
            oneIssue = IssueTemplate.model_validate_json(queryResult["documents"][0][resultIdx])

            oneIndexerQueryResult = OneIndexerQueryResult(
                score = distFloat,
                rank = resultIdx + 1,
                identifier = oneIssue.identifier,
                title = oneIssue.title,
                report = queryResult["metadatas"][0][resultIdx]["document"],
            )
            oneIndexerQueryResultList.appendQueryResult(
                identifier = oneIssue.identifier,
                queryResult = oneIndexerQueryResult
            )
        return oneIndexerQueryResultList

    
    def getDBJiraMatch(self, issueTemplate : BaseModel) -> ResultWithTypeList :
        """
        Query jira item collection and select vectors within cut-off distance
        
        Args:
            issueTemplate - issue from report table

        Returns:
            ResultWithTypeList
        """

        jiraWithTypeList = ResultWithTypeList(results_list = [])

        if hasattr(issueTemplate, "identifier"):
            queryString = issueTemplate.identifier
        else:
            msg = f"Jira Query cannot find identifier field"
            self.workerError(msg)
            return jiraWithTypeList

        queryResult = self._chromaJiraItems.query(query_texts=[queryString], n_results=1000)
        cutDist = 0.99
        resultIdx = -1
        for distFloat in queryResult["distances"][0] :
            resultIdx += 1
            if (distFloat > cutDist) :
                break
            # result from RAG issues table has attached typename for the conversion
            oneResultWithType = OneResultWithType(
                data = queryResult["documents"][0][0], 
                parser_typename = queryResult["metadatas"][0][0]["recordType"]
            )
            recordHash = hash(oneResultWithType)
            toAdd = True
            for existingItem in jiraWithTypeList.results_list:
                if recordHash == hash(existingItem):
                    toAdd = False
                    break
            if toAdd:
                jiraWithTypeList.results_list.append(oneResultWithType)

        if not len(jiraWithTypeList.results_list) :
            msg = f"Jira query {queryString} did not get matches less than {cutDist}"
            self.workerError(msg)
        return jiraWithTypeList


    def rrfReRanking(self, allQueryResults : AllIndexerQueryResults) -> AllIndexerQueryResults:
        """
        Reciprocal Rank Fusion (RRF) re-ranking of semantic and bm25s search results
        
        :param allQueryResults: query results
        :type allQueryResults: AllIndexerQueryResults
        :return: query results updated with rank
        :rtype: AllQueryResults
        """

    #    for item in allQueryResults.result_lists:
    #        msg = f"RRF:  {item.label} matches: {len(item.result_dict)}"    
    #        queryWorkflow.workerSnapshot(msg)

        # merge keys from all runs into set
        setKeys = set()
        for item in allQueryResults.listQueryResults:
            for key in item.result_dict:
                setKeys.add(key)

    #    msg = f"RRF: Length of combined keys: {len(setKeys)}"
    #    queryWorkflow.workerSnapshot(msg)
        
        # calculate rank for issue access all query runs
        scoresDict : Dict[str, IdentifierQueryResults] = {}
        for ident in list(setKeys):
            finalRank : float = 0.0
            oneQueryAppResult = None
            for item in allQueryResults.listQueryResults:
                if ident in item.result_dict:
                    oneQueryAppResult = item.result_dict[ident]
                    finalRank += 1/(60 + oneQueryAppResult.rank)
            identifierQueryResults = IdentifierQueryResults(
                identifier = ident,
                rrfRank = finalRank
            )
            scoresDict[ident] = identifierQueryResults
        # sort descending by RRF rank
        scoresDict = dict(sorted(scoresDict.items(), key=lambda item: item[1].rrfRank, reverse=True))
        allQueryResults.rrfScores = RRFScores(
            scoresDict = scoresDict
        )
        return allQueryResults


    def performQueries(self) -> AllIndexerQueryResults :

        allQueryResults = AllIndexerQueryResults(
            listQueryResults = [],
            rrfScores = {}
        )

        if not self.initRAGcomponents():
            return {}

        originalQuery  = self.query

        msg = f"original: {originalQuery}"
        self.workerSnapshot(msg)

        if QUERYTYPES.ORIGINAL in self.queryTransforms:
            if self.queryPreprocessFlag :
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed: {originalQuery}"
                self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(originalQuery, COLLECTION.ISSUES.value, "ORIG"))

        if QUERYTYPES.ORIGINALCOMPRESS in self.queryTransforms:
            if self.queryPreprocessFlag :
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed: {originalQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(originalQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "ORIGCOMPRESS"))

        if QUERYTYPES.HYDE in self.queryTransforms:
            hydeQuery = self.hydeQuery(originalQuery)
            msg = f"hyde: {hydeQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                hydeQuery = self.preprocessQuery(hydeQuery)
                msg = f"preprocessed: {hydeQuery}"
                self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(hydeQuery, COLLECTION.ISSUES.value, "HYDE"))

        if QUERYTYPES.HYDECOMPRESS in self.queryTransforms:
            hydeQuery = self.hydeQuery(originalQuery)
            msg = f"hyde: {hydeQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                hydeQuery = self.preprocessQuery(hydeQuery)
                msg = f"preprocessed: {hydeQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(hydeQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "HYDECOMPRESS"))

        if QUERYTYPES.MULTI in self.queryTransforms:
            multiQuery = self.multiQuery(originalQuery)
            msg = f"multi: {json.dumps(multiQuery)}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                multiQuery = self.preprocessQuery(multiQuery)
                msg = f"preprocessed: {multiQuery}"
                self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(multiQuery, COLLECTION.ISSUES.value, "MULTI"))

        if QUERYTYPES.MULTICOMPRESS in self.queryTransforms:
            multiQuery = self.multiQuery(originalQuery)
            msg = f"multi: {json.dumps(multiQuery)}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                multiQuery = self.preprocessQuery(multiQuery)
                msg = f"preprocessed: {multiQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(multiQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "MULTICOMPRESS"))

        if QUERYTYPES.REWRITE in self.queryTransforms:
            rewriteQuery = self.rewriteQuery(originalQuery)
            msg = f"rewrite: {rewriteQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                rewriteQuery = self.preprocessQuery(rewriteQuery)
                msg = f"preprocessed: {rewriteQuery}"
                self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(rewriteQuery, COLLECTION.ISSUES.value, "REWRITE"))

        if QUERYTYPES.REWRITECOMPRESS in self.queryTransforms:
            rewriteQuery = self.rewriteQuery(originalQuery)
            msg = f"rewrite: {rewriteQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                rewriteQuery = self.preprocessQuery(rewriteQuery)
                msg = f"preprocessed: {rewriteQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(rewriteQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.vectorQuery(rewriteQuery, COLLECTION.ISSUES.value, "REWRITECOMPRESS"))

        if QUERYTYPES.BM25SORIG in self.queryTransforms:
            if self.queryPreprocessFlag :
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed for BM25s: {originalQuery}"
                self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(originalQuery, self.queryBM25Options)
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.bm25sQuery(tokenizedQuery, self.bm25IndexFolder, "BM25SORIG"))

        if QUERYTYPES.BM25SORIGCOMPRESS in self.queryTransforms:
            if self.queryPreprocessFlag :
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed for TSC: {originalQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(originalQuery)
            msg = f"compressed for BM25s: {compressedQuery}"
            self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(compressedQuery, self.queryBM25Options)
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.bm25sQuery(tokenizedQuery, self.bm25IndexFolder, "BM25SORIGCOMPRESS"))

        if QUERYTYPES.BM25PREP in self.queryTransforms:
            bm25sQuery = self.prepBM25S(originalQuery)
            msg = f"prepared for BM25s: {bm25sQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                bm25sQuery = self.preprocessQuery(bm25sQuery)
                msg = f"preprocessed for BM25s: {bm25sQuery}"
                self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(bm25sQuery, self.queryBM25Options)
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.bm25sQuery(tokenizedQuery, self.bm25IndexFolder, "BM25SPREP"))

        if QUERYTYPES.BM25PREPCOMPRESS in self.queryTransforms:
            bm25sQuery = self.prepBM25S(originalQuery)
            msg = f"prepared for TSC: {bm25sQuery}"
            self.workerSnapshot(msg)
            if self.queryPreprocessFlag :
                bm25sQuery = self.preprocessQuery(bm25sQuery)
                msg = f"preprocessed for TSC: {bm25sQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(bm25sQuery)
            msg = f"compressed for BM25s: {compressedQuery}"
            self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(compressedQuery, self.queryBM25Options)
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.listQueryResults.append(self.bm25sQuery(tokenizedQuery, self.bm25IndexFolder, "BM25SPREPCOMPRESS"))

        allQueryResults = self.rrfReRanking(allQueryResults)
        return allQueryResults


    def threadWorker(self):
        """
        Workflow to perform query. 
        
        Args:
            None
        
        Returns:
            None

        """

        self._llmModel = self.createOpenAIModel()

        allQueryResults = self.performQueries()

        # output results files
        with open(self.outputFileName, "w") as jsonOut:
            jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

        msg = f"Processing completed."
        self.workerSnapshot(msg)
