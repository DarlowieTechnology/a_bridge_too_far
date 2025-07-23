#
# Read in JSON summary from job description
# Match products/technologies/role/certifications to known objects
#
import sqlite3
import sys
import tomli



def main():
    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return

    dictGlobalConfig = {}
    try:
        with open(sys.argv[1], mode="rb") as fp:
            dictGlobalConfig = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return

    with sqlite3.connect(dictGlobalConfig["database_name"]) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute('select * from skill')
        rows = cur.fetchall()
        result = [dict(row) for row in rows]

    print(f"number of rows: {len(rows)}")

    for row in rows:
        print(dict(row))
        print("---------")


if __name__ == "__main__":
    main()
