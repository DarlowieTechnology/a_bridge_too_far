#
# check if TABLE has empty slots - if so, fill the slots via LLM 
# (re)-create ChromeDB collection
# TABLE should match JSON file with format 
# { 
#   "list_of_records": [ 
#       { "id" = "", "name"=NAME, "description"=DESC }, ...
#   ] 
# }
#
import sys
import tomli

from typing import Union
from typing import List

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions
from pydantic import BaseModel, Field

import asyncio

# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile

#---------------------------------------------------


supportedCollections = ["scenario", "actreal"]


#---------------------------------------------------
# CAREFUL - does non-english
#

async def expandRecords(collectionName : str, allRecords : AllRecords) -> tuple[bool, Union[AllRecords, str]] :

    systemPrompt = f"""
        You are an expert in responsibilities of various organization roles.
        Your job is to expand on responsibility as supplied in the prompt
        """

    ollamaModel = OpenAIModel(
                        model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"])
                    )
    agent = Agent(ollamaModel, retries=1, output_retries=1, system_prompt = systemPrompt)

    idxRecord = -1
    for oneRecord in allRecords.list_of_records:

        idxRecord += 1

        oneRecord.id = str(idxRecord)

        if len(oneRecord.description):
#            print(f"Skipping completed record {idxRecord}")
            continue
        
        print(f"Expanding record {idxRecord}")

        prompt = f"""Output expanded paragraph for text in brackets. 
        Do not format text with newlines. Output only the answer. ( {oneRecord.name} )"""

        try:
            result = await agent.run(prompt)
            oneRecord.description = result.output
        except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
            print(f"ERROR: exception {e}")
            continue

    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], collectionName, allRecords)

    return True, allRecords


async def fillRecords(collectionName : str, allRecords : AllRecords) -> tuple[bool, Union[AllRecords, str]] :

    """Fill `id` field for all records. Fill `description` field if empty """

    idxRecord = -1
    for oneRecord in allRecords.list_of_records:
        idxRecord += 1
        oneRecord.id = str(idxRecord)
        if len(oneRecord.description):
            continue
        oneRecord.description = oneRecord.name.capitalize() + "."


    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], collectionName, allRecords)
    return True, allRecords



async def createCollection(chromaClient : chromadb.PersistentClient,
                     collectionName : str, 
                     allRecords : AllRecords) -> tuple[bool, AllRecords] : 
    """ create Chroma collection from record list """


    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url= ConfigSingleton().conf["rag_embed_url"],
    )

    chromaCollection = chromaClient.create_collection(
        name=collectionName,
        embedding_function=ef,
        metadata={ "hnsw:space": ConfigSingleton().conf["rag_hnsw_space"]  }
    )

    ids : list[str] = []
    docs : list[str] = []
    docMetadata : list[str] = []
    embeddings = []

    idxRecord = -1
    for oneRecord in allRecords.list_of_records:
        idxRecord += 1
        oneRecord.id = str(idxRecord)

        ids.append(oneRecord.id)
        docs.append(oneRecord.description)
        docMetadata.append({ "docName" : oneRecord.name } )
        if collectionName == "actreal":
            embeddings.append(ef([oneRecord.name])[0])
        if collectionName == "scenario":
            embeddings.append(ef([oneRecord.description])[0])
        if (idxRecord % 100) == 0 :
            print(f"added embedding {idxRecord}")

    chromaCollection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids,
        metadatas=docMetadata
    )

    print(f"ChromaDB collection {collectionName} created with {chromaCollection.count()} documents")
    return True, allRecords


async def updateCollection(chromaClient : chromadb.PersistentClient,
                     collectionName : str, 
                     allRecords : AllRecords) -> tuple[bool, AllRecords] : 
    """ update Chroma collection from record list """

    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url= ConfigSingleton().conf["rag_embed_url"],
    )

    chromaCollection = None
    try:
        chromaCollection = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        # Collection [TABLE] does not exists
        return await createCollection(chromaClient, collectionName, allRecords)

    # assume all ids in database are in the range 0..max
    # add only the records after max
    maxCnt = chromaCollection.count()
    recordsToAdd = []
    for oneRecord in allRecords.list_of_records:
        if oneRecord.id == str(maxCnt):
#            print(f"found record with id {maxCnt}")
            recordsToAdd = allRecords.list_of_records[maxCnt:]

#    print(f"maxCnt {maxCnt} number of elements to add {len(recordsToAdd)}")
    if len(recordsToAdd):
        ids : list[str] = []
        docs : list[str] = []
        docMetadata : list[str] = []
        embeddings = []

        idxRecord = maxCnt - 1
        for oneRecord in recordsToAdd:
            idxRecord += 1
            oneRecord.id = str(idxRecord)    

            ids.append(oneRecord.id)
            docs.append(oneRecord.description)
            docMetadata.append({ "docName" : oneRecord.name } )
            if collectionName == "actreal":
                embeddings.append(ef([oneRecord.name])[0])
            if collectionName == "scenario":
                embeddings.append(ef([oneRecord.description])[0])
            print(f"added embedding {oneRecord.id}")

        chromaCollection.add(
            embeddings=embeddings,
            documents=docs,
            ids=ids,
            metadatas=docMetadata
        )

    print(f"ChromaDB collection {collectionName} updated with {len(recordsToAdd)} documents")
    return True, allRecords


async def deleteCollection(chromaClient : chromadb.PersistentClient,
                     collectionName : str) -> bool : 
    """ delete Chroma collection"""
    print(f"Attempting to remove collection {collectionName}")
    try:
        chromaClient.delete_collection(name=collectionName)
        print(f"Collection {collectionName} removed")
    except Exception as e:
        # Exception Collection [TABLE] does not exists
        print(f"{e}")

    return True


async def main():

    if len(sys.argv) < 3:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG TABLE\nExample: {sys.argv[0]} default.toml scenario")
        return
    configName = sys.argv[1]
    try:
        with open(configName, mode="rb") as fp:
                ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {configName}, exception {e}")
        return
    chromaCollectionName = sys.argv[2]

    if chromaCollectionName not in supportedCollections :
        print(f"***ERROR: Collection {chromaCollectionName} is not supported")
        return

    boolResult, allRecordsOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], chromaCollectionName)
    if (not boolResult):
        print(allRecordsOrError)
        return
    print(f"Read total of {len(allRecordsOrError.list_of_records)} records from {chromaCollectionName} JSON")

#    boolResult, allRecordsOrError = await expandRecords(chromaCollectionName, allRecordsOrError)
#    if (not boolResult):
#        print(allRecordsOrError)
#        return

    boolResult, allRecordsOrError = await fillRecords(chromaCollectionName, allRecordsOrError)
    if (not boolResult):
        print(allRecordsOrError)
        return

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().conf["rag_datapath"],
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # await deleteCollection(chromaClient, chromaCollectionName)

    boolResult, allRecordsOrError = await updateCollection(chromaClient, chromaCollectionName, allRecordsOrError)
    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], chromaCollectionName, allRecordsOrError)

    return


    boolResult, chromaCollectionOrError = await createCollection(chromaClient, chromaCollectionName, allRecordsOrError)
    if not boolResult:
        print(chromaCollectionOrError)
        return


if __name__ == "__main__":
    asyncio.run(main())
