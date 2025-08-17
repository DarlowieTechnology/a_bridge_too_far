#
# create ChromeDB collection
# Accepts TOML configuration file and name of the collection
# name of the collection should match JSON file with format { "list_of_records": [ { "name"=NAME, "description"=DESC }, ...] }
#
import sys
import tomli

from typing import Union

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

import asyncio

# local
from common import AllRecords, ConfigSingleton, OpenFile

#---------------------------------------------------


supportedCollections = ["activity", "certifications", "position", "productcategory", "productfeature", "scenario"]


#---------------------------------------------------

def createCollection(chromaClient : chromadb.PersistentClient,
                     collectionName : str, 
                     jsonFileList : list[str]) -> tuple[bool, Union[chromadb.Collection, str]] : 
    """ create Chroma collection and embed from JSON files"""

    allRecords = AllRecords(list_of_records = [])
    for fileName in jsonFileList:
        boolResult, recordsOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], fileName)
        if not boolResult:
            return False, f"***ERROR: Cannot read file {fileName}  error: {recordsOrError}"
        
        print(f"Read total of {len(recordsOrError.list_of_records)} records from {fileName}")
        allRecords.list_of_records = allRecords.list_of_records + recordsOrError.list_of_records 

    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url= ConfigSingleton().conf["rag_embed_url"],
    )

    chromaCollection = chromaClient.create_collection(
        name=collectionName,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    ids : list[str] = []
    docs : list[str] = []
    docMetadata : list[str] = []
    embeddings = []

    idxRecord = -1
    for oneRecord in allRecords.list_of_records:
        idxRecord += 1
        label = "id" + str(idxRecord)

        ids.append(label)
        docs.append(oneRecord.description)
        docMetadata.append({ "docName" : oneRecord.name } )
        embeddings.append(ef([oneRecord.description])[0])
        print(f"added embedding {idxRecord}")

    chromaCollection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids,
        metadatas=docMetadata
    )

    print(f"ChromaDB collection {collectionName} created with {chromaCollection.count()} documents")
    return True, chromadb.Collection


async def main():

    if len(sys.argv) < 3:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG COLLECTION\nExample: {sys.argv[0]} default.toml scenario")
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
        print(f"***ERROR: Collection name {chromaCollectionName} is not supported")
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

    chromaCollection = None
    try:
        chromaCollection = chromaClient.get_collection(
            name=chromaCollectionName,
            embedding_function=ef
        )
        print(f"Existing collection {chromaCollectionName} opened with {chromaCollection.count()} documents")
    except Exception as e:
        print(f"Exception {e}")

        tableList = [chromaCollectionName]
        boolResult, chromaCollectionOrError = createCollection(chromaClient, chromaCollectionName, tableList)
        if not boolResult:
            print(chromaCollectionOrError)
            return

    if not chromaCollection:
        chromaCollection = chromaClient.get_collection(
            name=chromaCollectionName,
            embedding_function=ef
        )
        print(f"Collection opened with {chromaCollection.count()} documents")


if __name__ == "__main__":
    asyncio.run(main())
