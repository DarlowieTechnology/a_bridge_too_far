import sqlite3
import sys
import json
import glob
from pathlib import Path
import tomli

# local dependencies
#
from common import ConfigSingleton

#--------------Database creation script

statements = {
    "sector": {
        "create": """CREATE TABLE IF NOT EXISTS sector (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );""",
        "insert": """INSERT INTO sector(name)
              VALUES(:name);""",
    },
    "industry": {
        "create": """CREATE TABLE IF NOT EXISTS industry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                sector_name TEXT NOT NULL,
                FOREIGN KEY(sector_name) REFERENCES sector(name)
            );""",
        "insert": """INSERT INTO industry(name, sector_name)
              VALUES(:name, :sector_name);""",
    },
    "vendor": {
        "create": """CREATE TABLE IF NOT EXISTS vendor (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                industry_name TEXT NOT NULL,
                url TEXT,
                notes TEXT,
                started TEXT,
                ended TEXT,
                FOREIGN KEY(industry_name) REFERENCES industry(name)
            );""",
        "insert": """INSERT INTO vendor(name,industry_name,url,notes, started, ended)
              VALUES(:name, :industry_name, :url, :notes, :started, :ended);""",
    },
    "productcategory": {
        "create": """CREATE TABLE IF NOT EXISTS productcategory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                synonyms TEXT,                
                description TEXT
            );""",
        "insert": """INSERT INTO productcategory(name,synonyms,description)
              VALUES(:name, :synonyms, :description);""",
    },
    "productfeature": {
        "create": """CREATE TABLE IF NOT EXISTS productfeature (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                synonyms TEXT,
                description TEXT
            );""",
        "insert": """INSERT INTO productfeature(name,synonyms,description)
              VALUES(:name, :synonyms, :description);""",
    },
    "product": {
        "create": """CREATE TABLE IF NOT EXISTS product (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                feature_list TEXT,
                notes TEXT,
                FOREIGN KEY(category) REFERENCES productcategory(name),
                FOREIGN KEY(vendor) REFERENCES vendor(name)
            );""",
        "insert": """INSERT INTO product(vendor,category,name,feature_list,notes)
              VALUES( :vendor, :category, :name, :feature_list, :notes);""",
    },
    "position": {
        "create": """CREATE TABLE IF NOT EXISTS position (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            );""",
        "insert": """INSERT INTO position(name, description)
              VALUES(:name, :description);""",
    },
    "scenario": {
        "create": """CREATE TABLE IF NOT EXISTS scenario (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot TEXT NOT NULL,
                position TEXT NOT NULL,
                keywords TEXT NOT NULL
            );""",
        "insert": """INSERT INTO scenario(plot, position, keywords)
              VALUES(:plot, :position, :keywords);""",
    },
    "activity": {
        "create": """CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                position TEXT NOT NULL
            );""",
        "insert": """INSERT INTO activity(name, description, position)
              VALUES( :name, :description, :position);""",
    },
    "skill": {
        "create": """CREATE TABLE IF NOT EXISTS skill (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill TEXT NOT NULL,
                description TEXT,
                keyword_list TEXT 
            );""",
        "insert": """INSERT INTO skill(skill, description, keyword_list)
              VALUES( :skill, :description, :keyword_list);""",
    },
    "questions": {
        "create": """CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            );""",
        "insert": """INSERT INTO questions(question, answer)
              VALUES( :question, :answer);""",
    },
    "certifications": {
        "create": """CREATE TABLE IF NOT EXISTS certifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL
            );""",
        "insert": """INSERT INTO certifications(name, description)
              VALUES( :name, :description);""",
    }    
}


def createSchema(conn):
    cursor = conn.cursor()
    for key in statements:
        try:
            cursor.execute(statements[key]["create"])
            conn.commit()
        except sqlite3.OperationalError as e:
            print("***ERROR: Failed to create tables:", e)


#
# from supplied folder path read all JSON files
# insert records in tables with the same name
#
def insertData(conn):
    dataPath = ConfigSingleton().conf["sqlite_datapath"]
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        print(f"***ERROR: not a folder: {folder_path}")
        return

    # enable FOREIGN KEY constraint - off by default
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()

    files = glob.glob(dataPath + "/*.json")

    for key in statements:
        for file in files:
            file_path = Path(file)
            if not file_path.is_file():
                continue
            tableName = Path(file_path).stem

            # insert in the order of SQL statements
            if tableName == key:
                with open(file, "r") as recordFile:
                    try:
                        records = json.load(recordFile)
                    except json.JSONDecodeError as e:
                        print(f"***ERROR: file {file_path} is not a valid JSON: {e}")
                        return
                if len(records) == 0:
                    continue
                print(f"***Reading in {len(records)} from {file_path}")
                tableName = Path(file_path).stem
                for key in records:
                    dict = records[key]
                    cur = conn.cursor()
                    # catch and show UNIQUE, NOT NULL, FOREIGN KEY constraint errors
                    try:
                        cur.execute(statements[tableName]["insert"], dict)
                        conn.commit()
                    except sqlite3.IntegrityError as e:
                        print(f"sqlite3.IntegrityError: insert constraint failed: {dict}, exception {e}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return

    try:
        with open(sys.argv[1], mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return

    with sqlite3.connect(ConfigSingleton().conf["database_name"]) as conn:
        print(f"SQLite {sqlite3.sqlite_version}\nDatabase {ConfigSingleton().conf["database_name"]}")
        createSchema(conn)
        insertData(conn)

if __name__ == "__main__":
    main()
