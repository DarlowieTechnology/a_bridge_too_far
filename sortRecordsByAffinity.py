#
# read AllRecords from "RECORDS.json"
# get or create ChromaDB collection with embedding calculated by name field only
# for each record:
#   query ChromaDB by name field only, select 100 matches
#   merge all matches with distance under 0.07 into one record
# save set as AllRecords
  

import sys
import tomli
import json
from pathlib import Path
from typing import Union


import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile, DebugUtils

#----------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG TABLE\nExample: {sys.argv[0]} default.toml activity")
        return

    try:
        with open(sys.argv[1], mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return
    jsonName = sys.argv[2]


    boolResult, allRecordsOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonName)
    if (not boolResult):
        print(allRecordsOrError)
        return
    print(f"JSON {jsonName} opened with {len(allRecordsOrError.list_of_records)} records")

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().conf["rag_datapath"],
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url= ConfigSingleton().conf["rag_embed_url"],
    )

    try:
        chromaCollection = chromaClient.get_collection(
            name=jsonName,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        print("ERROR - create ChromaDB collection first!!!")
        return

    print(f"Collection {jsonName} opened with {chromaCollection.count()} documents")

    newRecordList = list(range(0, len(allRecordsOrError.list_of_records)))
    progressIndicator = 0
    sameDist = ConfigSingleton().conf["rag_same"]
    for oneRecord in allRecordsOrError.list_of_records :

        queryResult = chromaCollection.query(query_texts=[oneRecord.name], n_results=100)
        idx = -1
        for distFloat in queryResult["distances"][0] :
            if (distFloat >= sameDist) :
                break

            idx += 1
            docText = ""
            if (queryResult["documents"]) :
                docText = queryResult["documents"][0][idx]
            idText = queryResult["ids"][0][idx]

            if int(oneRecord.id) != int(idText) :
                print(f"Found same: '{oneRecord.id}' '{oneRecord.name}' --- '{idText}' '{docText}' --- '{distFloat}'")
                idToRemove = int(idText)
                if idToRemove < int(oneRecord.id):
                    idToRemove = int(oneRecord.id)
                try:
                    newRecordList.remove(idToRemove)
                    print(f"Removed id: '{idToRemove}'")
                except:
                    pass
                DebugUtils.pressKey()


        progressIndicator = progressIndicator +1
        if (progressIndicator % 100) == 0 :
            print(f"processed {progressIndicator}")        

    newAllRecords = AllRecords(list_of_records = [])
    for oneRecord in allRecordsOrError.list_of_records :
        if int(oneRecord.id) in newRecordList:
            newAllRecords.list_of_records.append(oneRecord)

    print(f"Original List {len(allRecordsOrError.list_of_records)} - updated list {len(newAllRecords.list_of_records)}")
    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonName, newAllRecords)


if __name__ == "__main__":
    main()
