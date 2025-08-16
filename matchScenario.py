#
# read vectors from chromadb/scenario collection - loose match 
# match some responsibilities with scenarios
#
from __future__ import annotations as _annotations

import sys
import tomli
import json
from pathlib import Path

from typing import Union
from typing import List

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb import QueryResult

import pydantic_ai
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import asyncio


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile


# ---------------data types

class OneDesc(BaseModel):
    """represents one description"""
    description: str = Field(..., description="description of records")

#---------------------------------------------------

def retrieve(search_query: str, chromaCollection: chromadb.Collection) -> QueryResult : 

    """Retrieve documentation based on a search query."""

    results = chromaCollection.query(
        query_texts=[search_query], 
            n_results=5
    )
    #print(type(results))
    #return results['metadatas'][0], results['documents'][0]
    return results

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
            name=ConfigSingleton().conf["rag_scenario_collection"],
            embedding_function=ef
        )
        print(f"Collection {ConfigSingleton().conf["rag_scenario_collection"]} opened with {chromaCollection.count()} documents")
    except Exception as e:
        print(f"{e}")
        return

    #    systemPrompt = f"""
    #        You are an expert in project descriptions. Create complete project description based on information supplied.
    #         """

    systemPromptComplex = f"""
        You are an expert resume creator. Create one paragraph description of the project for inclusion in resume. 
        Base on information supplied. 
        Here is the JSON schema for the OneRecord model you must 
        use as context for what information is expected:
        {json.dumps(OneRecord.model_json_schema(), indent=2)}
        """
    systemPromptSimple = f"""
        You are an expert resume creator. Create one paragraph description of the project for inclusion in resume. 
        Base on information supplied. 
        Here is the JSON schema for the OneDesc model you must 
        use as context for what information is expected:
        {json.dumps(OneDesc.model_json_schema(), indent=2)}
        """


    allRecords = AllRecords(list_of_records = [])
    inputToChromaDB = [
        "designing security controls for software applications",
        "implementing security controls for software applications",
        "maintaining security controls for software applications throughout their development lifecycle",
        "risk assessment",
        "secure coding practices",
        "threat modeling",
        "penetration testing",
        "vulnerability management",
        "incident response",
        "compliance with regulatory requirements",
        "ensuring CIA (confidentiality, integrity, and availability)",
        "minimizing attack surface through robust security architecture",
        "knowledge of application development technologies",
        "threat intelligence",
        "security standards and best practices",
        "regulatory frameworks",
        "collaboration with cross-functional teams including development, operations, QA for DevOps workflow"
        ]

    for oneInput in inputToChromaDB:
        chromaQuery = f"What are the scenarios for ({oneInput})?"
#       print(f"\n------------------------\nChromaDB query: ({chromaQuery})\n\n")

        dictResults = retrieve(chromaQuery, chromaCollection)
        cutoffDist = ConfigSingleton().conf["rag_distance"]
        idx = -1
        numberChosen = 0
        distList = []
        combinedDoco = ""
        for distFloat in dictResults["distances"][0] :
            idx += 1
            if (distFloat > cutoffDist) : 
                continue
            distList.append(distFloat)
            docText = ""
            if (dictResults["documents"]) :
                docText = dictResults["documents"][0][idx]
            metaInf = ""
            if (dictResults["metadatas"]) :
                metaInf = dictResults["metadatas"][0][idx]["docName"]
            combinedDoco = combinedDoco + "\n" + docText
            numberChosen = numberChosen + 1

        if not numberChosen :
            print("ERROR: cannot find ChromaDB records under distance cutoff")
            continue
#        else:
#            print(f"Selected {numberChosen} scenarios from ChromaDB. Distances: {distList}")

        ollamaModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                            provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))

        try:
            agent = Agent(ollamaModel,
                        output_type=OneRecord,
                        system_prompt = systemPromptComplex)

            prompt = f"""Here is the description of the project\n\n{combinedDoco}\n\nCreate a short paragraph"""
            print(f"\n\nagent prompt:\n---------\n{prompt}\n-------------\n\n")
            result = await agent.run(prompt)
            oneRecord = OneRecord.model_validate_json(result.output.model_dump_json())
            oneRecord.name = oneInput
#            print(oneRecord.model_dump_json(indent=2))
            allRecords.list_of_records.append(oneRecord)
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            print(f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior - retry")

            # try with simple type constrain
            try:
                agent = Agent(ollamaModel,
                            output_type=OneDesc,
                            system_prompt = systemPromptSimple)
                prompt = f"""Here is the description of the project\n\n{combinedDoco}\n\nCreate a short paragraph"""
                result = await agent.run(prompt)
                oneDesc = OneDesc.model_validate_json(result.output.model_dump_json())
                oneRecord = OneRecord(name=oneInput, description=oneDesc.description)
                allRecords.list_of_records.append(oneRecord)
            except Exception as e:
                print(f"Exception: {e} - give up")

        OpenFile.writeRecordJSON("shortprojects", allRecords)

if __name__ == "__main__":
    asyncio.run(main())
