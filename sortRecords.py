#
# read list of dict from "RECORDS.json"
# sort by name
# write "RECORDS.new.json"

import sys
import tomli
import json
from pathlib import Path

from typing import Union
from operator import itemgetter


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile

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

    allRecordsOrError.list_of_records = sorted(allRecordsOrError.list_of_records, key=lambda d: d.name)

    OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonName, allRecordsOrError)

if __name__ == "__main__":
    main()
