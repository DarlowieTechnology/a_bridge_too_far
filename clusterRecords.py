#
# read AllRecords from "RECORDS.json"
# get or create ChromaDB collection with embedding calculated by name field only
# for query in seed array
#   query ChromaDB by name field only
#   calculate number of close records

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

#----------------------------------------------

class OneVerbSubjectPair(BaseModel):
    """pair verb and subject for recombination"""
    verbs: List[str] = Field(..., description="list of verbs")
    subjects: List[str] = Field(..., description="list of subjects")
    current_verb: int = 0
    current_subject: int = -1

    def nextval(self) -> str:
        self.current_subject += 1
        if self.current_subject >= len(self.subjects):
            self.current_verb += 1
            if self.current_verb >= len(self.verbs):
                return None
            self.current_subject = 0

        verb = self.verbs[self.current_verb]
        subject = self.subjects[self.current_subject]
        return verb + " " + subject

class AllVerbSubjectPairs(BaseModel):
    """list of all verb and subject recombination"""
    list_of_pairs: List[OneVerbSubjectPair] = Field(..., description="list of pairs")
    current_pair:int = 0

    def nextval(self) -> str:
        if self.current_pair >= len(self.list_of_pairs):
            return None
        oneVerbSubjectPair = self.list_of_pairs[self.current_pair]
        val = oneVerbSubjectPair.nextval()
        if not val:
            self.current_pair += 1
            if self.current_pair >= len(self.list_of_pairs):
                return None
            oneVerbSubjectPair = self.list_of_pairs[self.current_pair]
            val = oneVerbSubjectPair.nextval()
        return val


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

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().conf["rag_datapath"],
        settings=Settings(anonymized_telemetry=False),
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
        # Collection [TABLE] does not exists
        print("ERROR - create ChromaDB collection first!!!")
        return

    print(f"Collection {jsonName} opened with {chromaCollection.count()} documents")

    queryArray = [
        AllVerbSubjectPairs(
            list_of_pairs = [
                OneVerbSubjectPair(
                    verbs = ["train", "coach", "mentor", "guide", "educate"],
                    subjects = ["colleagues", "employees", "users", "clients", "customers", "partners", "teams", "analysts"]
                ),
                OneVerbSubjectPair(
                    verbs = ["deliver", "develop", "provide", "conduct"],
                    subjects = ["training program", "training and guidance", "training sessions"]
                )
        ]),
        AllVerbSubjectPairs(
            list_of_pairs = [
                OneVerbSubjectPair(
                    verbs = ["document", "develop", "create", "review", "deliver", "maintain"],
                    subjects = ["plan", "policy", "procedure", "standard", "tool"]
                )
        ])
    ]

    allOutputRecords = AllRecords(list_of_records = [])
    allIdsOutputSet = set()

    for allVerbSubjectPairs in queryArray:

        allQueryResults = AllQueryResults(list_of_queryresults = [])
        allIdSet = set()
        seedQuery = allVerbSubjectPairs.nextval()
        while seedQuery:

            boolResult = makeQuery(chromaCollection, seedQuery, allQueryResults, allIdSet)
            if not boolResult:
                return
            seedQuery = allVerbSubjectPairs.nextval()

        print(f"total unique elements {len(allIdSet)} total elements {len(allQueryResults.list_of_queryresults)}")

        listIds = list(allIdSet)
        listIds.sort()
        for idRecord in listIds :
            if idRecord in allIdsOutputSet:
                continue
            for queryRecord in allQueryResults.list_of_queryresults:
                if int(queryRecord.id) == int(idRecord):
                    oneRecord = OneRecord(
                        id = str(idRecord),
                        name = queryRecord.name,
                        description = queryRecord.desc
                    )
                    allOutputRecords.list_of_records.append(oneRecord)
#                   DebugUtils.dumpPydanticObject(queryRecord, "Record from Query")
#                   if DebugUtils.pressKey():
#                        return
                    break
        allIdsOutputSet = allIdsOutputSet.union(allIdSet)

    for oneRecord in allRecordsOrError.list_of_records:
#        print(f"{oneRecord.id} {type(oneRecord.id)}  -- {type(allIdsOutputSet)}")
        if int(oneRecord.id) in allIdsOutputSet:
            pass
        else:
            allOutputRecords.list_of_records.append(OneRecord(
                        id = oneRecord.id,
                        name = oneRecord.name,
                        description = oneRecord.description
                    ))

    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonName, allOutputRecords)

def makeQuery(
        chromaCollection : chromadb.Collection, 
        seedQuery : str, 
        allQueryResults : AllQueryResults,
        allIdSet : set) -> bool :

    print(f"query ({seedQuery})")

    idSet = set()
    queryResult = chromaCollection.query(query_texts=[seedQuery], n_results=1000)

    idx = -1
    for distFloat in queryResult["distances"][0] :
        if (distFloat >= ConfigSingleton().conf["rag_cluster"]) :
            break
        idx += 1
        oneQueryResult = OneQueryResult(
                id = queryResult["ids"][0][idx],
                name = queryResult["metadatas"][0][idx]["docName"], 
                desc = queryResult["documents"][0][idx], 
                query = seedQuery,
                distance=distFloat 
            )        
        allQueryResults.list_of_queryresults.append(oneQueryResult)
        idSet.add(int(queryResult["ids"][0][idx]))

#        DebugUtils.dumpPydanticObject(oneQueryResult, "Record from Query")
#        if DebugUtils.pressKey():
#            return False
    print(f"found {len(idSet)} within distance {ConfigSingleton().conf['rag_cluster']}")

#    if not len(idSet):
#        # add potential record
#        with open("out.txt", "a") as fileOut:
#            fileOut.write(f'{{\n\t"id": "",\n\t"name": "{seedQuery}",\n\t"description": "{seedQuery}"\n}},\n')

    allIdSet = allIdSet.union(idSet)
    
    return True


if __name__ == "__main__":
    main()
