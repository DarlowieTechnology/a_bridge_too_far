#
# read list of dict from "RECORDS.json"
# sort by name
# write "RECORDS.new.json"

import sys
import logging
from pathlib import Path


# local
from common import ConfigSingleton, OpenFile

#----------------------------------------------

def main():

    scriptName = Path(sys.argv[0]).name

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(scriptName)

    if not (len(sys.argv) == 2):
        logger.info(f"Invalid number of arguments\nUsage:\n\t{scriptName} TABLE\nExample: {scriptName} activity")
        return

    config = ConfigSingleton()
    jsonName = sys.argv[1]

    boolResult, allRecordsOrError = OpenFile.readRecordJSON(config["sqlite_datapath"], jsonName)
    if (not boolResult):
        logger.info(allRecordsOrError)
        return

    allRecordsOrError.list_of_records = sorted(allRecordsOrError.list_of_records, key=lambda d: d.name)

    OpenFile.writeRecordJSON(config["sqlite_datapath"], jsonName, allRecordsOrError)

if __name__ == "__main__":
    main()
