#
# Query workflow class used by Django app and command line
#
import json
from typing import List, Dict
from typing_extensions import Self
from pathlib import Path
import time
import re
from pprint import pprint

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage

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
    query : list[str] = Field(default = [], description="List of queries for this workflow")
    queryTransforms : QUERYTYPES = Field(default = "", description="List of query transformation flags")
    bm25IndexFolder : str = Field(default = "", description="bm25 index folder")
    semanticMaxCutItemDistance : float = Field(default = 0.5, description = "Maximum distance in semantic search")
    semanticRetrieveNumber : int = Field(default = 512, description = "Number of items retrieved with semantic query")
    queryBM25Options : TOKENIZERTYPES = Field(default = TOKENIZERTYPES.STOPWORDSEN, description = "Options for BM25 query tokenizer")
    bm25sMinCutOffScore : float = Field(default = 0.0, description="Minimum bm25s score cut off")
    bm25sRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with bm25s query")
    outputNumber : int = Field(default = 1, description="Maximum number of items to return")
    outputFileName : str = Field(default = "", description="File name for results")
    queryPreprocessFlag : bool = Field(default = True, description="Call preprocessQuery() after every query transform")
    queryCompressFlag : bool = Field(default = True, description="Call Telegraphic Semantic Compression (TSC) after every query transform")

    stats : Dict[str, int] = Field(default = {}, description="Run statistics")

#    @model_validator(mode='after')
    def queryWorkflow_verify_configuration(self) -> Self:
        if not Path(self.dataFolder).is_dir:
            raise ValueError(f'Intermediate data folder is invalid')
        if not Path(self.bm25IndexFolder).is_dir:
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


    def preprocessQuery(self, queryList : list[str]) -> list[str] :
        """
        Preprocess string for interaction with LLM: convert all characters to ASCII, lowercase, remove whitespace, normalise spaces
        Can be used on user input or output of LLM

        :param queryList: original query 
        :type queryList: list[str]
        :return: preprocessed query
        :rtype: list[str]
        """
        prepQuery : list[str] = []
        for oneQuery in queryList:
            query = anyascii(oneQuery)
            query = query.strip().lower()
            query = re.sub(r'[^\w\s?!]', '', query)
            query = " ".join(query.split())
            prepQuery.append(query)
        return prepQuery


    def compressQuery(self, queryList : list[str]) -> list[str] :
        """
        Perform Telegraphic Semantic Compression (TSC) on the query for semantic search. 
        Ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/.
        Get english dictionary: python -m spacy download en_core_web_sm.

        :param queryList:  original query 
        :type queryList: list[str]
        :return: compressed query
        :rtype: list[str]
        """

        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Parts of speech to remove (predictable grammar)
        REMOVE_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}

        # Optional low-information words to remove
        REMOVE_LIKE = {"like", "just", "really", "basically", "literally"}

        compressedQuery : list[str] = []

        for query in queryList:
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
            compressedQuery.append(newQuery)

        return compressedQuery

    
    def tokenizeQuery(self, queryList : list[str], tokenizerTypes: TOKENIZERTYPES) -> list[str]:
        """
        create tokens from the BM25S search.
        
        :param queryList: original query
        :type queryList: list[str]
        :param tokenizerTypes: flags for tokenizer
        :type tokenizerTypes: TOKENIZERTYPES
        :return: list of tokens
        :rtype: list[str]
        """

        if TOKENIZERTYPES.STOPWORDSEN in tokenizerTypes:
            stopWords = "english"
        else:
            stopWords = None

        if TOKENIZERTYPES.STEMMER in tokenizerTypes:
            stemmer=Stemmer.Stemmer("english")
        else:
            stemmer = None

        query_tokens = bm25s.tokenize(queryList, return_ids = False, stopwords = stopWords, stemmer = stemmer)
        return query_tokens


    def hydeQuery(self, queryList : list[str]) -> list[str]:
        """ 
        Use HyDE (Hypothetical Document Embedding) to improve the query for semantic search. 

        :param queryList:  original query 
        :type queryList: list[str]
        :return: HyDE query
        :rtype: list[str]
        """

        startHyDE = time.time()
        systemPrompt = f"Write a two sentence answer to the user prompt query"
        agentHyDE = Agent(self._llmModel, system_prompt = systemPrompt)

        queryHyDE : list[str] = []

        for query in queryList:
            userPrompt = query
            try:
                result = agentHyDE.run_sync(userPrompt)
                resultQuery = result.output
                if result.usage():
                    self.addUsage(result.usage())
                    self.updateStats(topKey = "HyDE", keyValList = [("Count", 1), ("Time", time.time() - startHyDE), ("Input Tokens", result.usage().input_tokens), ("Output Tokens", result.usage().output_tokens)])
                else:
                    self.updateStats(topKey = "HyDE", keyValList = [("Count", 1), ("Time", time.time() - startHyDE)])
                queryHyDE.append(resultQuery)
            except Exception as e:
                self.updateStats(topKey = "HyDE", keyValList = [("Exception", 1)])
        return queryHyDE


    def multiQuery(self, queryList : list[str]) -> list[str]:
        """
        Generate multiple queries form the original query for semantic search. 

        :param queryList:  original query 
        :type queryList: list[str]
        :return: Multi query
        :rtype: list[str]
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

        queryMultiple : list[str] = []

        for query in queryList:
            userPrompt = query
            try:
                result = agentMultipleQ.run_sync(userPrompt)
                resultQuery = result.output
                if result.usage():
                    self.addUsage(result.usage())
                    self.updateStats(topKey = "Multi", keyValList = [("Count", 1), ("Time", time.time() - startMulti), ("Input Tokens", result.usage().input_tokens), ("Output Tokens", result.usage().output_tokens)])
                else:
                    self.updateStats(topKey = "Multi", keyValList = [("Count", 1), ("Time", time.time() - startMulti)])
                queryMultiple.append(resultQuery)
            except Exception as e:
                self.updateStats(topKey = "Multi", keyValList = [("Exception", 1)])

        return queryMultiple


    def rewriteQuery(self, queryList : list[str]) -> list[str]:
        """
        Rewrite the query 

        :param queryList:  original query 
        :type queryList: list[str]
        :return: Rewrite query
        :rtype: list[str]
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
        agentRewriteQ = Agent(self._llmModel, system_prompt = systemPrompt)

        queryRewrite : list[str] = []

        for query in queryList:
            userPrompt = query
            try:
                result = agentRewriteQ.run_sync(userPrompt)
                resultQuery = result.output
                if result.usage():
                    self.addUsage(result.usage())
                    self.updateStats(topKey = "Rewrite", keyValList = [("Count", 1), ("Time", time.time() - startRewrite), ("Input Tokens", result.usage().input_tokens), ("Output Tokens", result.usage().output_tokens)])
                else:
                    self.updateStats(topKey = "Rewrite", keyValList = [("Count", 1), ("Time", time.time() - startRewrite)])
                queryRewrite.append(resultQuery)
            except Exception as e:
                self.updateStats(topKey = "Rewrite", keyValList = [("Exception", 1)])

        return queryRewrite


    def prepBM25S(self, queryList : list[str]) -> list[str]:
        """Prepare query for BM25S search using LLM. 

        :param queryList:  original query 
        :type queryList: list[str]
        :return: BM25S query
        :rtype: list[str]
        """

        # Prompt for generating prepared query
        systemPrompt = """You are a cybersecurity expert. 
        Return a list of terms in the original user prompt with their descriptions.
        Return only the results. Format output as a list of Python strings.
        Original query is supplied in user prompt"""

        startPrepBM25s = time.time()
        agentPrepBM25s = Agent(self._llmModel, system_prompt = systemPrompt)

        queryBM25SPrep : list[str] = []

        for query in queryList:
            userPrompt = query
            try:
                result = agentPrepBM25s.run_sync(userPrompt)
                resultQuery = result.output
                if result.usage():
                    self.addUsage(result.usage())
                    self.updateStats(topKey = "PrepBM25S", keyValList = [("Count", 1), ("Time", time.time() - startPrepBM25s), ("Input Tokens", result.usage().input_tokens), ("Output Tokens", result.usage().output_tokens)])
                else:
                    self.updateStats(topKey = "PrepBM25S", keyValList = [("Count", 1), ("Time", time.time() - startPrepBM25s)])
                queryBM25SPrep.append(resultQuery)
            except Exception as e:
                self.updateStats(topKey = "prepBM25S", keyValList = [("Exception", 1)])

        return queryBM25SPrep


    def bm25sQuery(self, queryList : list[str], folderName : str, queryLabel : str) -> OneIndexerQueryResultList : 
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

        startBM25sQuery = time.time()
        oneIndexerQueryResultList = OneIndexerQueryResultList(
            query = queryList[0],
            label = queryLabel
        )

        retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)

        max_items = self.bm25sRetrieveNumber
        if retriever.scores["num_docs"] < self.bm25sRetrieveNumber:
            max_items = retriever.scores["num_docs"]

        results, scores = retriever.retrieve(queryList, k=max_items)
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
                identifier = docN[1].strip() + "|" + docN[0].strip()
                oneIndexerQueryResultList.appendQueryResult(
                    identifier = identifier,
                    queryResult = oneIndexerQueryResult
                )

        self.updateStats(topKey = "PrepBM25S", keyValList = [("Count", 1), ("Time", time.time() - startBM25sQuery), ("Items", max_items)])
        return oneIndexerQueryResultList


    def vectorQuery(self, queryList : list[str], collection : COLLECTION, queryLabel : str) -> OneIndexerQueryResultList:
        """
        Performs vector (semantic) query. Returns list of results
        Uses semanticMaxCutItemDistance to cut results off
        Uses semanticRetrieveNumber to limit max number of items
        
        :param queryList: query for semantic search
        :type queryList: list[str]
        :param collection: chroma DB collection name for query
        :type query: COLLECTION
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :return: list of results
        :rtype: OneIndexerQueryResultList
        """
        startVectorQuery = time.time()
        oneIndexerQueryResultList = OneIndexerQueryResultList(
            query = queryList,
            label = queryLabel
        )

        chromaCollection = self.collections[collection]

        queryResult = chromaCollection.query(query_texts = queryList, n_results = self.semanticRetrieveNumber)

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
            identifier = oneIssue.title + "|" + oneIssue.identifier
            oneIndexerQueryResultList.appendQueryResult(
                identifier = identifier,
                queryResult = oneIndexerQueryResult
            )
        self.updateStats(topKey = "Vector", keyValList = [("Count", 1), ("Time", time.time() - startVectorQuery), ("Items", resultIdx + 1)])
        return oneIndexerQueryResultList

    
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

        # merge ident keys from all runs into set
        setKeys = set()
        for item in allQueryResults.listQueryResults:
            for key in item.result_dict:
                setKeys.add(key)

    #    print(f"=========\n{setKeys}\n==============")
    #    msg = f"RRF: Length of combined keys: {len(setKeys)}"
    #    queryWorkflow.workerSnapshot(msg)
        
        # calculate rank for issue access all query runs
        scoresDict : Dict[str, IdentifierQueryResults] = {}
        for ident in list(setKeys):
            finalRank : float = 0.0
            for item in allQueryResults.listQueryResults:
                if ident in item.result_dict:
                    itemRank = item.result_dict[ident].rank
                    finalRank += 1/(60 + itemRank)
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
            query = self.query,
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
        totalStart = time.time()

        self._llmModel = self.createOpenAIModel()

        allQueryResults = self.performQueries()

        # output results files
        with open(self.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

        self.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart), ("Usage", self.totalUsageFormat(insertHTML = False) ) ])
        pprint(self.stats)
