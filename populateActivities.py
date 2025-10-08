# Extract data from job ads
#
# pre-requisite
#   both activity and scenario are loaded to ChromaDB and match JSON files. Use "createChrome.py" to fill ChromaDB.
#
# read all job ads in "jobDescriptions"
# use two step process to extract "activities, technologies, methodologies, software services, and software products"
# match each item to "actreal" in ChromaDB conf["rag_distance"] - 0.2
# if not match - output potential activity to "out.txt"
# Arguments:
#  CONFIG - default.toml configuration file
# Example call:
#    populateActivities.py default.toml
# Output:
#    out.txt - unmatched activity records
# Notes:
#   Script can be used interactively (uncomment break points) or batch. 
#   Script will re-create file "out.txt" on every run.
# 


import sys
import tomli
import json
import re
import sqlite3
from pathlib import Path

from typing import Union

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile, DebugUtils

#--------------------------------------------------------

class OneResultList(BaseModel):
    """represents one results list from LLM call"""
    results_list: list[str] = Field(..., description="list of results")

class LLMresult(BaseModel):
    """represents collection of results from LLM calls"""
    originFile: str = ""
    dict_of_results: dict[str, OneResultList]

#-------------------------------------------------------


def main():
    
    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return
    try:
        with open(sys.argv[1], mode="rb") as fp:
                ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return

    # get list of job ads
    boolResult, listFilePathsOrError = OpenFile.readListOfFileNames("jobDescriptions", "*.txt")
    if (not boolResult):
        print(listFilePathsOrError)
        return
    print(f"Read in {len(listFilePathsOrError)} input files names")

    # read in job ads
    allJobAdRecords = AllRecords(list_of_records=[])
    for jobDescriptionPath in listFilePathsOrError:
        boolResult, contentJDOrError = OpenFile.open(filePath = jobDescriptionPath, readContent = True)
        if not boolResult:
            print(contentJDOrError)
            continue
        oneRecord = OneRecord(id = "", name=str(jobDescriptionPath), description=contentJDOrError)
        allJobAdRecords.list_of_records.append(oneRecord)
    if (not len(allJobAdRecords.list_of_records)) : 
        print(f"***ERROR: Cannot read Job Descriptions")
        return
    print(f"Read in {len(allJobAdRecords.list_of_records)} Job Descriptions")

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().conf["rag_datapath"],
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    chromaActions = getChromaCollection(chromaClient, "actreal")
    if not chromaActions:
        return

    chromaScenario = getChromaCollection(chromaClient, "scenario")
    if not chromaScenario:
        return

    # remove stale content from "out.txt"
    with open("out.txt", "w") as fileOut:
        pass

    # process job ads
    for jobAddRecord in allJobAdRecords.list_of_records:

        print(f"Processing {jobAddRecord.name}")

        llmResult = LLMresult(originFile = str(jobDescriptionPath), dict_of_results = {})

        oneResultList = extractInfo(jobAddRecord)
        if not oneResultList:
            continue
        itemSet = set()
        for itm in oneResultList.results_list:
            listNew = getChromaDBMatchActivity(chromaActions, itm)
            for itemNew in listNew.results_list:
                itemSet.add(itemNew)
        oneResultList = OneResultList(results_list = list(itemSet))
        DebugUtils.dumpPydanticObject(oneResultList, "Mapped list")




#        if oneResultList:
#            llmResult.dict_of_results["name"] = oneResultList
#        llmResult.dict_of_results["questions"] = processQuestions(jobDescription = contentJDOrError)
#        jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobDescriptionPath).stem + ".json"
#        with open(jobDescriptionJSON, "w") as jsonOut:
#            res = llmResult.model_dump_json(indent=2)
#            jsonOut.writelines(res)


def extractInfo(jobInfo : OneRecord) -> OneResultList :

    systemPromptPhase1 = f"""
        You are an expert in cyber security, information technology and software development 
        You will be supplied text of job advertisement.
        Your job is to extract information from the text that matches user's request.
        """

    ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollModel, 
                system_prompt = systemPromptPhase1,
                retries=10,
                output_retries=10)

    promptPhase1 = f"""Extract the list of 
    activities, technologies, methodologies, software services, and software products from the text below.
    Combine all items in common list of strings.
    Do not separate by category.
    Avoid single word items.
    Output only lower-case characters.
    Output only the result.
    \n
    {jobInfo.description}
            """

    try:
        result = agent.run_sync(promptPhase1)
        print(f"{result.output}")
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        print(f"extractInfo: Skipping due to exception: {e}")
        return None

    phase2Input = result.output

    systemPromptPhase2 = f"""
        You are an expert in JSON processing.
        Input prompt contains list in JSON format. 

        Here is the JSON schema for the OneResultList model you must use as context for what information is expected:
        {json.dumps(OneResultList.model_json_schema(), indent=2)}
        """
    promptPhase2 = f"""
                {phase2Input}
            """

    agentPhase2 = Agent(ollModel, 
                output_type=OneResultList,
                system_prompt = systemPromptPhase2,
                retries=20,
                output_retries=20)
    try:
        result = agentPhase2.run_sync(promptPhase2)
        oneResultList = OneResultList.model_validate_json(result.output.model_dump_json())
        DebugUtils.dumpPydanticObject(oneResultList, "Phase 2")
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        print(f"extractInfo: Skipping due to exception: {e}")
        return None
    return oneResultList


# Seek adds question in known fixed format after this tag:
# ^Employer questions$
# ^Your application will include the following questions:$
#
def processQuestions(jobDescription:str) -> OneResultList : 
    oneResult = OneResultList(results_list = [])
    pattern = r'^Employer questions.*\nYour application will include the following questions:.*\n'
    matchQuestion = re.search(pattern, jobDescription, re.MULTILINE)
    if matchQuestion:
        questionsStr = jobDescription[matchQuestion.end(0):]
        oneResult.results_list = questionsStr.splitlines(False)
    return oneResult


def getChromaDBMatchActivity(chromaDBCollection : Collection, queryString : str) -> OneResultList :

    totals = set()

    queryResult = chromaDBCollection.query(query_texts=[queryString], n_results=1)
    cutDist = ConfigSingleton().conf["rag_distance"]
    resultIdx = -1
    for distFloat in queryResult["distances"][0] :
        resultIdx += 1
        docText = ""
        if (queryResult["documents"]) :
            docText = queryResult["documents"][0][resultIdx]

        if (distFloat > cutDist) :
            break

        totals.add(docText)

    if not len(totals) :

        with open("out.txt", "a") as fileOut:
            fileOut.write(f'{{\n\t"id": "","name": "{queryString}",\n\t"description": "{queryString.capitalize()}."\n}},')
        return OneResultList(results_list = [])

    return OneResultList(results_list=list(totals))


#
# open chromeDB Collection
#
def getChromaCollection(chromaClient : chromadb.PersistentClient, collName : str) -> Collection :

    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url=ConfigSingleton().conf["rag_embed_url"]    
    )

    try:
        chromaColl = chromaClient.get_collection(
            name=collName,
            embedding_function=ef
        )
        print(f"Collection {collName} opened with {chromaColl.count()} documents")
    except Exception as e:
        print(f"Exception: {e}")
        return None
    return chromaColl



if __name__ == "__main__":
    main()
