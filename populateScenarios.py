# Utility to fill scenarios from activities
#
# pre-requisite
#   both activity and scenario are loaded to ChromaDB and match JSON files. Use "createChrome.py" to fill ChromaDB.
#
# read and match activity and scenario. If activity is not matched - (distance conf["rag_scenario"] or higher) record potential scenario in "out.txt"
# Data Schema:
#    "activity from job ad" -> "activity name" -> "scenario description" 
# where:
#    "activity from ad" - parsed from job ad. Recorded as "name" field in activity JSON
#    "activity name" - Recorded as "description" field in activity JSON and as "name" field in scenario JSON
#    "scenario description" - Recorded as "description" field in scenario JSON.
# Arguments:
#  CONFIG - default.toml configuration file
#  TABLE1 - activity table name (actreal)
#  TABLE2 - scenario table name (scenario)
#  COUNT - limits how many activity records the script will read. 
#          Useful, when you want to process only new records that you place on the top of activity table
# Example call:
#    populateScenarios.py default.toml actreal scenario 1000000
# Output:
#    out.txt - unmatched activity records
# Notes:
#   Script can be used interactively (uncomment break points) or batch. 
#   Script will re-create file "out.txt" on every run.
# 

import sys
import logging
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# local
from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile, DebugUtils

def main():

    scriptName = Path(sys.argv[0]).name

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(scriptName)

    if not (len(sys.argv) == 3):
        logger.info(f"\nUsage:\n\t{scriptName} TABLE1 TABLE2 COUNT\nExample: {scriptName} activity scenario 100")
        return

    config = ConfigSingleton()
    nameActivity = sys.argv[1]
    nameScenario = sys.argv[2]
    maxCount = sys.argv[3]

    boolResult, allActivityRecordsOrError = OpenFile.readRecordJSON(config["sqlite_datapath"], nameActivity)
    if (not boolResult):
        logger.info(allActivityRecordsOrError)
        return
    logger.info(f"JSON {nameActivity} opened with {len(allActivityRecordsOrError.list_of_records)} records")

    boolResult, allScenarioRecordsOrError = OpenFile.readRecordJSON(config["sqlite_datapath"], nameScenario)
    if (not boolResult):
        logger.info(allScenarioRecordsOrError)
        return
    logger.info(f"JSON {nameScenario} opened with {len(allScenarioRecordsOrError.list_of_records)} records")

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
        activityCollection = chromaClient.get_collection(
            name=nameActivity,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        logger.info(f"ERROR - create ChromaDB collection for {nameActivity} first!!!")
        return

    logger.info(f"Collection {nameActivity} opened with {activityCollection.count()} documents")

    try:
        scenarioCollection = chromaClient.get_collection(
            name=nameScenario,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        logger.info(f"ERROR - create ChromaDB collection for {nameScenario} first!!!")
        return

    logger.info(f"Collection {nameScenario} opened with {scenarioCollection.count()} documents")

    # remove stale content from "out.txt"
    with open("out.txt", "w") as fileOut:
        pass

    matchCount = 0
    recordIdx = 0
    notMatched = 0

    for actRecord in allActivityRecordsOrError.list_of_records:
        allQueryResults = AllQueryResults(list_of_queryresults = [])
        actDesc = actRecord.description
        queryResult = scenarioCollection.query(query_texts=actDesc, n_results=1000)

        idx = -1

        for distFloat in queryResult["distances"][0] :
            idx += 1
            
            if (distFloat > config["rag_scenario"]) :
                break

            oneResult = OneQueryResult(
                    id = queryResult["ids"][0][idx],
                    name = queryResult["metadatas"][0][idx]["docName"], 
                    desc = queryResult["documents"][0][idx], 
                    query = actDesc,
                    distance=distFloat 
                )

            allQueryResults.list_of_queryresults.append(oneResult)

# uncomment for interactive 
#            logger.info(f"\nid: {actRecord.id}   Query: ({actDesc})\n")
#            DebugUtils.dumpPydanticObject(oneResult, "one query result")
#            if DebugUtils.pressKey("Press c to move to next activity:"):
#                break

        if len(allQueryResults.list_of_queryresults):
            matchCount += 1

# uncomment to dump all matched scenarios
#            logger.info(f"({actDesc}) matched {len(allQueryResults.list_of_queryresults)} scenarios")
#            DebugUtils.dumpPydanticObject(allQueryResults, "Matched scenarios")

        else:
            logger.info(f"----------\n({actDesc}) not matched\n---------------")
            notMatched += 1

            # add potential scenario record
            with open("out.txt", "a") as fileOut:
                fileOut.write(f'{{\n\t"id": ""\n\t,"name": "{actRecord.description}",\n\t"description": "{actRecord.description}"\n}},')

        recordIdx += 1
        if recordIdx > int(maxCount):
            break
        if (recordIdx % 100) == 0 :
            logger.info(f"processed {recordIdx}")        

    logger.info(f"Distance {config['rag_scenario']}  Matches {matchCount}")
    logger.info(f"Not matched under {config['rag_scenario']} distance {notMatched}")


if __name__ == "__main__":
    main()
