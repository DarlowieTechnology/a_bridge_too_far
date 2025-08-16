#
# read position.json - strict description of positions
# read vectors from chromadb/position collection - loose match 
# extract responsibilities from position description
# write "responsibilities.json" 
#
from __future__ import annotations as _annotations

import os

import sys
import tomli
import json
from pathlib import Path

from typing import List

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

import pydantic_ai
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent


import asyncio


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile


# ---------------data types

class Responsibilities(BaseModel):
    """ responsibilities for employees role in organization """
    name: str = Field(..., description="Name of employees role")
    responsibilities: List[str] = Field(..., description="list of responsibilities")

class AllResponsibilities(BaseModel):
    """represents collection of all Responsibilities records"""
    list_of_records: List[Responsibilities] = Field(..., description="List of Responsibilities.")

#---------------------------------------------------

def writeResponsibilitiesJSON(allResponsibilities : AllResponsibilities) -> bool:
    """write all updated activity to new activity file"""

    dataPath = ConfigSingleton().conf["sqlite_datapath"]
    folder_path = Path(dataPath)
    jsonFileName = str(folder_path) + '/responsibilities.new.json'
    with open(jsonFileName, "w") as jsonOut:
        jsonOut.writelines(allResponsibilities.model_dump_json(indent=2))
    return True

def retrieve(search_query: str, chromaCollection: chromadb.Collection) -> tuple[str, str] : 

    """Retrieve documentation based on a search query."""

    results = chromaCollection.query(
        query_texts=[search_query], 
            n_results=1
    )
    #print(results)
    return results['metadatas'][0][0]["docName"], results['documents'][0][0]



async def main():
    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return
    configName = sys.argv[1]
    try:
        with open(configName, mode="rb") as fp:
                ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {configName}, exception {e}")
        return

    boolResult, positionsOrError = OpenFile.readRecordJSON("position")
    if not boolResult:
        print(f"***ERROR: Cannot read file position  error: {positionsOrError}")
        return False

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().conf["rag_datapath"],
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url=ConfigSingleton().conf["rag_embed_url"]    
    )

    try:
        chromaCollection = chromaClient.get_collection(
            name="position",
            embedding_function=ef
        )
        print(f"Collection position opened with {chromaCollection.count()} documents")
    except Exception as e:
        print(f"{e}")
        return
    
    systemPrompt = f"""
        You are an expert in responsibilities of various organization roles.
        Your job is to extract responsibilities from the the role definition. 
        
        Here is the JSON schema for the Responsibilities model you must 
        use as context for what information is expected:
        {json.dumps(Responsibilities.model_json_schema(), indent=2)}
        """

    ollamaModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollamaModel,
                output_type=Responsibilities,
                system_prompt = systemPrompt)

    allResponsibilities = AllResponsibilities(list_of_records = [])

#    for oneDict in positionsOrError.list_of_records :
#      positionName = oneDict.name
#      chromaQuery = f"What are the responsibilities of {positionName}?"
#      print(positionName)

    inputPositionName = "Senior specialist in implementing business continuity plans"
    chromaQuery = f"What are the responsibilities of {inputPositionName}?"

    databasePositionName, fullDesc = retrieve(chromaQuery, chromaCollection)

    prompt = f"""Here is the description of the role of {databasePositionName}.\n\n
                {fullDesc}\n\n
                Extract responsibilities."""
    print(f"agent prompt:\n\n{prompt}")

    try:
        result = await agent.run(prompt)
#            result.output.model_dump_json(indent=2)
        resplObj = Responsibilities.model_validate_json(result.output.model_dump_json())
        print(resplObj.model_dump_json(indent=2))
#        allResponsibilities.list_of_records.append(resplObj)
    except pydantic_ai.exceptions.UnexpectedModelBehavior:
        print(f"Skipping due to exception: []")

#    writeResponsibilitiesJSON(allResponsibilities)

if __name__ == "__main__":
    asyncio.run(main())
