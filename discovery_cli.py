import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4
from typing import List

from pydantic_ai.usage import RunUsage

import json_schema_to_pydantic
from anyascii import anyascii

# local
import darlowie
from common import COLLECTION, RecordCollection, MatchingSections, SectionInfo, AllTopicMatches
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
    discoveryWorkflow.context["stage"] = "started"


#    fileList = discoveryWorkflow.formFileList()
    fileListEngineering = [
        "documents/engineering-000.txt",
        "documents/engineering-001.txt",
        "documents/engineering-002.txt",
        "documents/engineering-003.txt",
        "documents/engineering-004.txt",
        "documents/engineering-005.txt",
        "documents/engineering-006.txt",
        "documents/engineering-007.txt",
        "documents/engineering-008.txt",
        "documents/engineering-009.txt",
        "documents/engineering-010.txt",
        "documents/engineering-011.txt",
        "documents/engineering-012.txt",
        "documents/engineering-013.txt",
        "documents/engineering-014.txt",
        "documents/engineering-015.txt",
        "documents/engineering-016.txt",
        "documents/engineering-017.txt",
        "documents/engineering-018.txt",
        "documents/engineering-019.txt"
    ]

    fileList = [
#        "documents/1904.10509v1.pdf",
#        "documents/1912.02292v1.pdf",
#        "documents/1912.06680v1.pdf",
#        "documents/2005.00341v1.pdf",
#        "documents/2005.14165v4.pdf",
        "documents/2009.03393v1.pdf"
#        "documents/2102.12092v2.pdf",
#       "documents/2103.00020v1.pdf",
    ]

    msg = f"Discovered {len(fileList)} files for processing"
    discoveryWorkflow.workerSnapshot(msg)

    totalCounts = [0] * 4
    sections = []

    for inputFileName in fileList:
        counts, allTopicMatches = discoveryWorkflow.processOneFile(inputFileName)
        totalCounts[0] += counts[0]
        totalCounts[1] += counts[1]
        totalCounts[2] += counts[2]
        totalCounts[3] += counts[3]
        for key in allTopicMatches.topic_dict.keys():
            matchingSections = allTopicMatches.topic_dict[key]
            for section in matchingSections.section_list:
                sections.append(section)


    score = totalCounts[0] - totalCounts[1] - totalCounts[2] * 0.5
    if score < 0:
        score = 0
    if totalCounts[3] > 0:
        scorePerCent = (score/totalCounts[3]) * 100
    else:
        scorePerCent = 0

    for section in sections:
        discoveryWorkflow.workerSnapshot(str(section))


    # ---------------completed ---------------

    print(f"totalCounts: {totalCounts}    score:{scorePerCent:.2f} %")

    totalEnd = time.time()
    discoveryWorkflow.context["stage"] = "completed"
    msg = f"Workflow completed. {discoveryWorkflow.totalUsageFormat()}. Total time {(totalEnd - totalStart):.2f} seconds."
    discoveryWorkflow.workerSnapshot(msg)



def main():

    context = darlowie.context

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["DISCLIsession_key"])


    context["documentFolder"] = "documents"
    context["fileExtensions"] = ["*.txt", "*.pdf"]

    context["status"] = []
    context["statusFileName"] = context["DISCLIstatus_FileName"]

    context["loadDocument"] = True
    context["parseChunks"] = True
    context["matchChunks"] = True
    context["vectorize"] = True
    context["verify"] = False

    # text extraction configuration from PDF
    context["stripWhiteSpace"] = False
    context["convertToLower"] = False
    context["convertToASCII"] = False
    context["singleSpaces"] = False

    discoverWorkflow = DiscoveryWorkflow(context, logger)
    testRun(discoverWorkflow)


if __name__ == "__main__":
    main()
