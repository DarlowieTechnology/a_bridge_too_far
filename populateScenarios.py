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
import tomli
import json
from pathlib import Path
from typing import Union
from typing import List
from typing import Sequence

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pydantic import BaseModel, Field


# local
from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile, DebugUtils

def main():
    if len(sys.argv) < 4:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG TABLE1 TABLE2 COUNT\nExample: {sys.argv[0]} default.toml activity scenario 100")
        return

    try:
        with open(sys.argv[1], mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return
    nameActivity = sys.argv[2]
    nameScenario = sys.argv[3]
    maxCount = sys.argv[4]

    boolResult, allActivityRecordsOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], nameActivity)
    if (not boolResult):
        print(allActivityRecordsOrError)
        return
    print(f"JSON {nameActivity} opened with {len(allActivityRecordsOrError.list_of_records)} records")

    boolResult, allScenarioRecordsOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], nameScenario)
    if (not boolResult):
        print(allScenarioRecordsOrError)
        return
    print(f"JSON {nameScenario} opened with {len(allScenarioRecordsOrError.list_of_records)} records")

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
        activityCollection = chromaClient.get_collection(
            name=nameActivity,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        print(f"ERROR - create ChromaDB collection for {nameActivity} first!!!")
        return

    print(f"Collection {nameActivity} opened with {activityCollection.count()} documents")

    try:
        scenarioCollection = chromaClient.get_collection(
            name=nameScenario,
            embedding_function=ef
        )
    except chromadb.errors.NotFoundError as e:
        print(f"ERROR - create ChromaDB collection for {nameScenario} first!!!")
        return

    print(f"Collection {nameScenario} opened with {scenarioCollection.count()} documents")

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
            
            if (distFloat > ConfigSingleton().conf["rag_scenario"]) :
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
#            print(f"\nid: {actRecord.id}   Query: ({actDesc})\n")
#            DebugUtils.dumpPydanticObject(oneResult, "one query result")
#            if DebugUtils.pressKey("Press c to move to next activity:"):
#                break

        if len(allQueryResults.list_of_queryresults):
            matchCount += 1

# uncomment to dump all matched scenarios
#            print(f"({actDesc}) matched {len(allQueryResults.list_of_queryresults)} scenarios")
#            DebugUtils.dumpPydanticObject(allQueryResults, "Matched scenarios")

        else:
            print(f"----------\n({actDesc}) not matched\n---------------")
            notMatched += 1

            # add potential scenario record
            with open("out.txt", "a") as fileOut:
                fileOut.write(f'{{\n\t"id": ""\n\t,"name": "{actRecord.description}",\n\t"description": "{actRecord.description}"\n}},')

        recordIdx += 1
        if recordIdx > int(maxCount):
            break
        if (recordIdx % 100) == 0 :
            print(f"processed {recordIdx}")        

    print(f"Distance {ConfigSingleton().conf["rag_scenario"]}  Matches {matchCount}")
    print(f"Not matched under {ConfigSingleton().conf["rag_scenario"]} distance {notMatched}")


if __name__ == "__main__":
    main()
