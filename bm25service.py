#
# BM25 service class used by Indexer and Discovery
#
from typing import List
from pathlib import Path
import json
import hashlib
import bm25s
from pydantic import BaseModel, Field

# local
import darlowie
from common import OneDiscoveryBM25File, DiscoveryBM25FileList, OpenFile

class Bm25Service(BaseModel):

    indexFolder: Path = Field(..., description="path to bm25 index folder")

    def bm25CreateCorpus(self, fileList: List[str]) -> List[str] :
        """
        Create corpus from all chunks

        :param fileList: list of input files
        :type fileList: List[str]
        :return: corpus in JSONL format
        :rtype: List[str]
        
        """

        # make path for bm25 index folder if does not exist
        self.indexFolder.mkdir(parents=True, exist_ok=True)

        # read list of files in bm25 index
        fileNameList = str(self.indexFolder) + "/corpus_files.json"
        boolResult, contentOrError = OpenFile.open(filePath = fileNameList, readContent = True)
        if boolResult:
            jsonStr = json.loads(contentOrError)
            discoveryBM25FileList = DiscoveryBM25FileList.model_validate(jsonStr)
        else:
            discoveryBM25FileList = DiscoveryBM25FileList(file_dict = {})

        # check if file was processed before

        for fileName in fileList:
            relPath = darlowie.context["DISCOVdocumentFolder"] + fileName
            boolResult, contentOrError = OpenFile.open(filePath = relPath, readContent = True)
            if boolResult:
                if fileName in discoveryBM25FileList.file_dict:
                    metaRec = discoveryBM25FileList.file_dict[fileName]
                    hashFunc = hashlib.sha256()
                    hashFunc.update(contentOrError.encode('utf-8'))
                    hashOnFile = hashFunc.hexdigest()
                    if (hashOnFile == metaRec.hash):
                        print(f"{fileName} - EXISTS - Same Version")
                    else:
                        print(f"{fileName} - EXISTS - New Version")
                        print(f"{hashOnFile}\n{metaRec.hash}")
                else:
                    print(f"{fileName} - NOT EXISTS")

        corpus = []

        return corpus


    def bm25sProcessCorpus(self, corpus : list[str], folderName: str) -> List[List[str]] :
        """
        Tokenize corpus
        Store bm25s index in a folder

        Args:
            corpus (list[str]) - list of strings representing identifier and title of all issues across documents
            folderName (str) - name of folder to save bm25s index
        Returns:
            bm25s compatible index
        """

#        stemmer = Stemmer.Stemmer("english")
 #       corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(folderName)
        return corpus_tokens


def main():

    folderName = Path(darlowie.context["DISCOVdocumentFolder"] + darlowie.context["DISCOVbm25IndexFolder"])
    bm25Service = Bm25Service(indexFolder = folderName)

    fileList = [
        "1904.10509v1.pdf",
        "1912.02292v1.pdf",
        "1912.06680v1.pdf",
        "2005.00341v1.pdf",
        "2005.14165v4.pdf",
        "2009.03393v1.pdf"
        "2102.12092v2.pdf",
       "2103.00020v1.pdf",
    ]

    bm25Service.bm25CreateCorpus(fileList)


if __name__ == "__main__":
    main()

