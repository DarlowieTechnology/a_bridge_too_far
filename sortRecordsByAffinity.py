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
import logging


import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile, DebugUtils

#----------------------------------------------

def main():

    scriptName = Path(sys.argv[0]).name
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(scriptName)

    if not (len(sys.argv) == 2):
        logger.info(f"\nUsage:\n\t{scriptName} TABLE\nExample: {scriptName} activity")
        return

    config = ConfigSingleton()
    jsonName = sys.argv[2]


    boolResult, allRecordsOrError = OpenFile.readRecordJSON(config["sqlite_datapath"], jsonName)
    if (not boolResult):
        logger.info(allRecordsOrError)
        return
    logger.info(f"JSON {jsonName} opened with {len(allRecordsOrError.list_of_records)} records")

    chromaClient = chromadb.PersistentClient(
        path=config["rag_datapath"],
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    ef = OllamaEmbeddingFunction(
        model_name=config["rag_embed_llm"],
        url= config["rag_embed_url"],
    )

    try:
        chromaCollection = chromaClient.get_collection(
            name=jsonName,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        logger.info("ERROR - create ChromaDB collection first!!!")
        return

    logger.info(f"Collection {jsonName} opened with {chromaCollection.count()} documents")

    newRecordList = list(range(0, len(allRecordsOrError.list_of_records)))
    progressIndicator = 0
    sameDist = config["rag_same"]
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
                logger.info(f"Found same: '{oneRecord.id}' '{oneRecord.name}' --- '{idText}' '{docText}' --- '{distFloat}'")
                idToRemove = int(idText)
                if idToRemove < int(oneRecord.id):
                    idToRemove = int(oneRecord.id)
                try:
                    newRecordList.remove(idToRemove)
                    logger.info(f"Removed id: '{idToRemove}'")
                except:
                    pass
                DebugUtils.pressKey()


        progressIndicator = progressIndicator +1
        if (progressIndicator % 100) == 0 :
            logger.info(f"processed {progressIndicator}")        

    newAllRecords = AllRecords(list_of_records = [])
    for oneRecord in allRecordsOrError.list_of_records :
        if int(oneRecord.id) in newRecordList:
            newAllRecords.list_of_records.append(oneRecord)

    logger.info(f"Original List {len(allRecordsOrError.list_of_records)} - updated list {len(newAllRecords.list_of_records)}")
    OpenFile.writeRecordJSON(config["sqlite_datapath"], jsonName, newAllRecords)


if __name__ == "__main__":
    main()
