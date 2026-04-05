import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4
from typing import List
import threading


from pydantic_ai.usage import RunUsage
from anyascii import anyascii

# local
import darlowie
from common import COLLECTION, ConfigCollection, RecordCollection, AllTopicMatches
from discovery_workflow import DiscoveryWorkflow




def testRun(discoveryWorkflow : DiscoveryWorkflow) -> list[str]:
    """ 
    Test for discovery app
    
    Args:
        discoveryWorkflow (DiscoveryWorkflow) - workflow object
    Returns:
        None    
    """
    totalStart = time.time()
    discoveryWorkflow.stage = "started"


    fileList = discoveryWorkflow.formFileList()
    fileListEngineering = [
        "documents/engineering-000.json",
        "documents/engineering-001.json",
        "documents/engineering-002.json",
        "documents/engineering-003.json",
        "documents/engineering-004.json",
        "documents/engineering-005.json",
        "documents/engineering-006.json",
        "documents/engineering-007.json",
        "documents/engineering-008.json",
        "documents/engineering-009.json",
        "documents/engineering-010.json",
        "documents/engineering-011.json",
        "documents/engineering-012.json",
        "documents/engineering-013.json",
        "documents/engineering-014.json",
        "documents/engineering-015.json",
        "documents/engineering-016.json",
        "documents/engineering-017.json",
        "documents/engineering-018.json",
        "documents/engineering-019.json"
    ]

    fileListMedical = [
        "documents/medresearch-000.txt",
        "documents/medresearch-001.txt",
        "documents/medresearch-002.txt",
        "documents/medresearch-003.txt",
        "documents/medresearch-004.txt",
        "documents/medresearch-005.txt",
        "documents/medresearch-006.txt",
        "documents/medresearch-007.txt",
        "documents/medresearch-008.txt",
        "documents/medresearch-009.txt",
        "documents/medresearch-010.txt",
        "documents/medresearch-011.txt",
        "documents/medresearch-012.txt",
        "documents/medresearch-013.txt",
        "documents/medresearch-014.txt",
        "documents/medresearch-015.txt",
        "documents/medresearch-016.txt",
        "documents/medresearch-017.txt",
        "documents/medresearch-018.txt",
        "documents/medresearch-019.txt"
    ]

    fileListLLM = [
        "documents/1904.10509v1.pdf",
        "documents/1912.02292v1.pdf",
        "documents/1912.06680v1.pdf",
        "documents/2005.00341v1.pdf",
        "documents/2005.14165v4.pdf",
        "documents/2009.03393v1.pdf",
        "documents/2102.12092v2.pdf",
        "documents/2103.00020v1.pdf",
        "documents/2107.03374v2.pdf",
        "documents/2110.05448v1.pdf",
        "documents/2112.10741v3.pdf",
        "documents/2202.01344v1.pdf",
        "documents/2212.04356v1.pdf",
        "documents/2303.01469v2.pdf",
        "documents/2303.08774v6.pdf",
        "documents/2305.20050v1.pdf",
        "documents/2312.09390v1.pdf",
        "documents/2406.04093v1.pdf",
        "documents/2410.21276v1.pdf",
        "documents/2412.16720v1.pdf"
    ]

    fileListPenTest = [
        "documents/AWS_Review.pdf",
        "documents/CD_and_DevOps Review.pdf",
        "documents/Database Review.pdf",
        "documents/Firewall Review.pdf",
        "documents/phpMyAdmin.pdf",
        "documents/PHP_Code_Review.pdf",
        "documents/Refinery-CMS.pdf",
        "documents/WASPT_Report.pdf",
        "documents/Web App and Ext Infrastructure Report.pdf",
        "documents/Web App and Infrastructure and Mobile Report.pdf",
        "documents/Wikimedia.pdf"
    ]

    fullFileList = fileListEngineering + fileListMedical + fileListLLM + fileListPenTest

#    fileList = [
#        "documents/2412.16720v1.pdf"
#        "documents/AWS_Review.pdf"
#        "documents/Database Review.pdf", # three 500, Time: 420.24 seconds
#        "documents/2009.03393v1.pdf"  # five tries, Time: 75.62 seconds - 63 chunks - exceed max retries
#        "documents/1912.02292v1.pdf"
#    ]

    msg = f"Discovered {len(fileList)} files for processing."
    discoveryWorkflow.workerSnapshot(msg)

    totalCounts = [0] * 4
    chunks = []

    for inputFileName in fileList:
        counts, allTopicMatches = discoveryWorkflow.processOneFile(inputFileName)
        totalCounts[0] += counts[0]
        totalCounts[1] += counts[1]
        totalCounts[2] += counts[2]
        totalCounts[3] += counts[3]
        for key in allTopicMatches.topic_dict.keys():
            matchingChunks = allTopicMatches.topic_dict[key]
            for chunk in matchingChunks.chunk_list:
                chunks.append(chunk)


    score = totalCounts[0] - totalCounts[1] - totalCounts[2] * 0.5
    if score < 0:
        score = 0
    if totalCounts[3] > 0:
        scorePerCent = (score/totalCounts[3]) * 100
    else:
        scorePerCent = 0

    for chunk in chunks:
        discoveryWorkflow.workerSnapshot(str(chunk))


    # ---------------completed ---------------

    msg = f"TotalCounts: {totalCounts}    score:{scorePerCent:.2f} %"
    discoveryWorkflow.workerSnapshot(msg)

    with open("fails.json", "w" , encoding="utf-8", errors="ignore") as jsonOut:
        jsonOut.writelines(json.dumps(discoveryWorkflow.getFails(), indent=2))

    totalEnd = time.time()
    discoveryWorkflow.stage = "completed"
    msg = f"Workflow completed. {discoveryWorkflow.totalUsageFormat()}. Total time {(totalEnd - totalStart):.2f} seconds."
    discoveryWorkflow.workerSnapshot(msg)


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    context = darlowie.context

    context["status"] = []
    context["statusFileName"] = context["DISCLIstatus_FileName"]

    # workflow actions
    context["loadDocument"] = False
    context["parseChunks"] = False
    context["makeRawVector"] = False
    context["bm25Process"] = False
    context["matchChunks"] = True
    context["verify"] = False
    context["returnResults"] = False
    context["clear"] = False

    # text extraction configuration
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    # other app-specific configuration
    context["documentFolder"] = "documents"
    context["fileExtensions"] = ["*.txt", "*.pdf", "*.json"]
    context["chunkSize"] = 256
    context["chunkOverlap"] = 48

    # search configuration
    context["searchSemanticOriginal"] = True
    context["searchBM25sOriginal"] = True
    context["searchSemanticMulti"] = True
    context["searchBM25sMulti"] = True
    context["searchSemanticRewrite"] = True
    context["searchBM25sRewrite"] = True
    context["searchSemanticHyDE"] = True
    context["searchBM25sHyDE"] = True

    # retrieval configuration
    context["semanticRetrieveNumber"] = 10
    context["semanticMaxCutItemDistance"] = 1.0
    context["bm25sRetrieveNumber"] = 10
    context["bm25sMinCutOffScore"] = 0.001
    context["rrfCutOffValue"] = 0.000001
    context["rrfOutlierZScoreThreshold"] = 3
    context["outputNumber"] = 5

    configCollection = ConfigCollection(context)

    discoverWorkflow = DiscoveryWorkflow()
    discoverWorkflow.configure(configCollection)

#    testRun(discoverWorkflow)

    thread = threading.Thread( target=discoverWorkflow.threadWorker)
    thread.start()
    thread.join()



if __name__ == "__main__":
    main()
