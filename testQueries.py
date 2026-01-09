from enum import Enum, IntEnum, unique,auto

from typing import Dict, List, Any

from pydantic import BaseModel, Field

import numpy as np


from resultsQueryClasses import SEARCH, OneQueryAppResult, OneQueryResultList, AllQueryResults

@unique
class TESTSET(IntEnum) :
    NOTEST = 0
    XSS = auto()
    CREDS = auto()


class TestSetCollection(object):
    """
    represents collection of test sets for query (XSS, etc.)
    """
    allSets_dict : Dict[TESTSET, List[str]] = {}
    current_test : TESTSET = TESTSET.NOTEST

    def __new__(cls):
        """ overwrite of __new__ to enforce one instance via class attribute 'instance' """
        if not hasattr(cls, 'instance'):
            cls.instance = super(TestSetCollection, cls).__new__(cls)
        return cls.instance 

    def getCurrentTest(self):
        if self.current_test not in self.allSets_dict:
            self.allSets_dict[self.current_test] = TestQuery(self.current_test, knowIssuesDict[self.current_test])
        return self.allSets_dict[self.current_test]
    
    def getCurrentTestType(self):
        return self.current_test

    def getCurrentTestName(self) ->str:
        if self.current_test == TESTSET.NOTEST:
            return 'None'
        if self.current_test == TESTSET.XSS:
            return 'XSS issues'
        if self.current_test == TESTSET.CREDS:
            return 'Credentials issues'

        return self.current_test

    def setCurrentTest(self, test : TESTSET):
        self.current_test = test


# known XSS issues 
knowIssuesDict = {
    TESTSET.XSS : [
        "outdated jenkins software\nsr-128-2-3",                                # CD_and_DevOps Review.pdf
        "phpmyadmin content security policy\ntestorg-1604_moss_phpmyadmin-002", # phpMyAdmin.pdf
        "self xss in table_row_action.php\ntestorg-1604_moss_phpmyadmin-008",   # phpMyAdmin.pdf
        "stored xss in the title text in the image upload\npt-rcms-001",        # Refinery-CMS.pdf
        "stored xss in the alt text in the image upload\npt-rcms-002",          # Refinery-CMS.pdf
        "stored xss in refinery cms add new page title\npt-rcms-003",           # Refinery-CMS.pdf
        "reflected cross-site scripting\nsr-101-003",                           # Web App and Ext Infrastructure Report.pdf
        "reflected cross-site scripting (xss)\nsr-102-001",                     # Web App and Infrastructure and Mobile Report.pdf
        "reflected xss in api.php\ntstorg-wmf1214-8",                           # wikimedia.pdf
        "stored xss in uploaded svg files\ntstorg-wmf1214-11",                  # wikimedia.pdf
        "stored xss in pdf files\ntstorg-wmf1214-14",                           # wikimedia.pdf
        "custom javascript may yield privilege escalation\ntstorg-wmf1214-10",  # wikimedia.pdf
        "users can inspect each other's personal javascript\ntstorg-wmf1214-7"  # wikimedia.pdf
    ],
    TESTSET.CREDS : [
        "finding weak administrative credentials\ntstorg-examplecorp001-011",                   # Architecture Review - Threat Model Report.pdf.json
        "use of shared administrator credentials\ntstorg-examplecorp001-002",                   # Architecture Review - Threat Model Report.pdf.json
        "weak password complexity requirements\ntstorg-examplecorp001-009",                     # Architecture Review - Threat Model Report.pdf.json
        "user without mfa enabled\nsr-109-004",                                                 # AWS_Review.pdf
        "iam password policy\nsr-109-005",                                                      # AWS_Review.pdf
        "credentials in github repository\nsr-128-1-2",                                         # CD_and_DevOps Review.pdf
        "clear text credentials\nsr-128-2-5",                                                   # CD_and_DevOps Review.pdf
        "iam password policy\nsr-128-3-2",                                                      # CD_and_DevOps Review.pdf
        "default passwords\nsr-103-003",                                                        # Database Review.pdf
        "default profile settings not configured\nsr-103-005",                                  # Database Review.pdf
        "enforce password expiration not set\nsr-103-015",                                      # Database Review.pdf
        "public role permissions on xp_instance_regread and xp_regread stored procedures\nsr-103-017", # Database Review.pdf
        "hard-coded trivial credentials\ntestorgphp-002",                                       # PHP_Code_Review.pdf
        "multiple complex authentication methods\ntestorg-samplewapt-009",                      # WASPT_Report.pdf
        "hard-coded credentials\nsr-102-009",                                                   # Web App and Infrastructure and Mobile Report.pdf
        "lack of upper limit on password length allows dos\ntstorg-wmf1214-1",                  # wikimedia.pdf
        "weak password policy\ntstorg-wmf1214-2"                                                # wikimedia.pdf
    ]
}

 

#----------------------ListComparisonResult-------------------------------------

class ListComparisonResult[T] :
    """
    represents comparison between two generic lists
    """
    list1 : List[str]
    Obj2 : Any = None

    def __init__(self, l1: List[str]) -> None:
        self.list1 = l1

    def inKnownSet(self, item : T) ->bool:
        title = item.title.lower()
        ident = item.identifier.lower()
        for val in self.list1:
            val = val.splitlines()
            if title == val[0] and ident == val[1]:
                return True
#        print(f"NOTFOUND: {title}|{val[0]}    {ident}|{val[1]}")
        return False

    def common(self) -> List[str]:
        commonItems = set()
        for item1 in self.list1:
            item1List = item1.splitlines()
            title1 = item1List[0].lower()
            ident1 = item1List[1].lower()
            for item2 in self.Obj2.result_dict:
                title2 = (self.Obj2.result_dict[item2]).title.lower()
                ident2 = item2.lower()
                if title1 == title2 and ident1 == ident2:
                    commonItems.add(item1)
                    break
        return list(commonItems)

    def firstonly(self) -> List[str]:
        firstOnly = set()
        for item1 in self.list1:
            item1List = item1.splitlines()
            title1 = item1List[0].lower()
            ident1 = item1List[1].lower()
            found = False
            for item2 in self.Obj2.result_dict:
                title2 = (self.Obj2.result_dict[item2]).title.lower()
                ident2 = item2.lower()
                if title1 == title2 and ident1 == ident2:
                    found = True
                    break
            if not found:
                firstOnly.add(item1)
        return list(firstOnly)

    def secondonly(self) -> List[str]:
        secondOnly = set()
        for item2 in self.Obj2.result_dict:
            title2 = (self.Obj2.result_dict[item2]).title.lower()
            ident2 = item2.lower()
            found = False
            for item1 in self.list1:
                item1List = item1.splitlines()
                title1 = item1List[0].lower()
                ident1 = item1List[1].lower()
                if title1 == title2 and ident1 == ident2:
                    found = True
                    break
            if not found:
                missedItem = f"{title2}\n{ident2}"
                secondOnly.add(missedItem)
        return list(secondOnly)


#----------------------StatsOnList--------------------------------------------

class StatsOnList(BaseModel) :
    """represents statistics of data set"""
    length : float = Field(default=0, description="length of dataset")
    min : float = Field(default=0, description="minimum value in the dataset") 
    max : float = Field(default=0, description="maximum value in the dataset") 
    avg : float = Field(default=0, description="average value in the dataset") 
    mean : float = Field(default=0, description="mean value in the dataset") 
    median : float = Field(default=0, description="median value in the dataset") 
    range : float = Field(default=0, description="range of values in the dataset") 
    q1 : float = Field(default=0, description="1st quartile of the dataset") 
    q2 : float = Field(default=0, description="2nd quartile of the dataset") 
    q3 : float = Field(default=0, description="3nd quartile of the dataset") 


    def makeStatsOnList(self, scoresForStats) :
        """
        Calculate stats on the list of float
        
        :param scoresForStats: list of float
        """

        if not len(scoresForStats):
            return

        a1F = np.array(scoresForStats, dtype=np.float32)

        self.length = len(a1F)
        self.min = np.min(a1F)
        self.max = np.max(a1F)
        self.avg = np.average(a1F)
        self.mean = np.mean(a1F)
        self.median = np.median(a1F)
        self.range = np.max(a1F)-np.min(a1F)
        self.q1, self.q2, self.q3 = np.quantile(a1F, [0.25, 0.5, 0.75])



#------------------------------TestQuery-----------------------------------------

class TestQuery():
    """
    Applies collection of known tests to the results of query
    """

    testset : TESTSET  = Field(..., description="type of test set")
    knownIssues : List[str] = Field(..., description="list of all known issues")
    missing : List[str] = Field(..., description="list of known issues missing from test results")
    common : List[str] = Field(..., description="list of issues common in known list and test results list")
    extra : List[str] = Field(..., description="list of issues in test results that are not in known issues")

    def __init__(self, testset : TESTSET, knownIssueList : List[str]):
        self.testset = testset
        self.knownIssues = knownIssueList
        super().__init__()


    def inKnowsSet(self, oneQueryAppResult : OneQueryAppResult) -> bool:
        """
        Test to check if the issue is in known test set
        
        :param oneQueryAppResult: query result
        :type oneQueryAppResult: OneQueryAppResult
        :return: True if the issue is in known test set, False otherwise
        :rtype: bool
        """
        return ListComparisonResult(self.knownIssues).inKnownSet(oneQueryAppResult)


    def testResultList(self, oneQueryResultList : OneQueryResultList) :
        """
        Check result list against known test set. Output statistics.
        
        :param oneQueryResultList: list of query results
        :type oneQueryResultList: OneQueryResultList
        :return: number of items not found.
        :rtype: int
        """

        compareObject = ListComparisonResult(self.knownIssues)
        compareObject.Obj2 = oneQueryResultList
        self.missing = compareObject.firstonly()
        self.common = compareObject.common()
        self.extra = compareObject.secondonly()


    def getListStats(self, oneQueryResultList : OneQueryResultList) -> tuple[StatsOnList, StatsOnList]:
        compareObject = ListComparisonResult(self.knownIssues)
        scoresForStats = []
        scoresForKnownSet = []
        for key in oneQueryResultList.result_dict:
            oneQueryAppResult = oneQueryResultList.result_dict[key]
            scoresForStats.append(oneQueryAppResult.score)
            if compareObject.inKnownSet(oneQueryAppResult):
                scoresForKnownSet.append(oneQueryAppResult.score)

        statsOnListGlobal = StatsOnList()
        statsOnListGlobal.makeStatsOnList(scoresForStats)

        statsOnListKnown = StatsOnList()
        statsOnListKnown.makeStatsOnList(scoresForKnownSet)

        return statsOnListGlobal, statsOnListKnown


    def outputRunInfo(self, oneQueryResultList : OneQueryResultList, label : str) -> str:

        compareObject = ListComparisonResult(self.knownIssues)
        compareObject.Obj2 = oneQueryResultList
        self.missing = compareObject.firstonly()
        self.common = compareObject.common()
        self.extra = compareObject.secondonly()
        statsOnListGlobal, statsOnListKnown = self.getListStats(oneQueryResultList)
        strOut = (
            f"{label}:GLOBAL  Q1: {statsOnListGlobal.q1:.4f}, Q2: {statsOnListGlobal.q2:.4f}, Q3: {statsOnListGlobal.q3:.4f} length: {statsOnListGlobal.length}, mean: {statsOnListGlobal.mean:.4f}, range: {statsOnListGlobal.range:.4f}"
            f"{label}:KNOWN length: {statsOnListKnown.length}, mean: {statsOnListKnown.mean:.4f}, range: {statsOnListKnown.range:.4f} , M|C|E: {len(self.missing)}|{len(self.common)}|{len(self.extra)}"
        )
        return strOut


    def calculateOverallScore(self, allQueryResults : AllQueryResults, maxResults : int) -> float:
        """
        Calculate overall score for set of query results. Score range is [0:1]
        100% score criteria: all known issues are top rank. 
        Each gap is scored negative 0.1 * (1/len(self.knownIssues))
        iteration stops at maxResults
        Missing issues scored each negative 1/len(self.knownIssues)
        
        :param allQueryResults: query results
        :type allQueryResults: AllQueryResults
        :param maxResults: max number of results to display
        :type maxResults: int
        :type allQueryResults: AllQueryResults
        :return: Overall score as a float number between 0 (fail) and 1 (perfect) 
        :rtype: float
        """
        compareObject = ListComparisonResult(self.knownIssues)
        knownFound = 0
        score = 1.0
        delta = 1/len(self.knownIssues)
        idx = 0
        for ident in allQueryResults.rrfScores:
            rank, oneResult = allQueryResults.rrfScores[ident]
            if compareObject.inKnownSet(oneResult):
                knownFound += 1
            else:
                score -= delta * 0.1
            if score <= 0.0:
                break
            if knownFound >= len(self.knownIssues):
                break
            idx += 1
            if idx >= maxResults:
                break
        if knownFound < len(self.knownIssues):
            score = score - (len(self.knownIssues) - knownFound) * delta
        if score < 0.0:
            score = 0.0
        allQueryResults.overall_score = score
        return score


    def outputRRFInfo(self, rrfScores : Dict[str, List[tuple[int, OneQueryAppResult]]], maxResults : int) -> List[str]:
        compareObject = ListComparisonResult(self.knownIssues)
        idx = 0
        oneQueryResultList = OneQueryResultList(
            result_dict = {},
            query = "",
            searchType = SEARCH.BM25S.value,
            label = ""            
        )
        outStrings = []
        for ident in rrfScores:
            rank, oneResult = rrfScores[ident]
            if compareObject.inKnownSet(oneResult):
                msg = f"+ {rank:.4f} {oneResult.title.lower()} ({ident.lower()})"
                print(msg)
                outStrings.append(msg)
                oneQueryResultList.appendResult(identifier=ident.lower(), title = oneResult.title.lower(), report = "", score = 0.0, rank = 0)
            else:
                msg = f"- {rank:.4f} {oneResult.title.lower()} ({ident.lower()})"
                print(msg)
                outStrings.append(msg)
            idx += 1
            if idx >= maxResults:
                break
        compareObject.Obj2 = oneQueryResultList
        missing = compareObject.firstonly()
        if len(missing):
            msg = "<b>Missing known items</b>"
            print(msg)
            outStrings.append(" ")
            outStrings.append(msg)
            outStrings.append(" ")
            for item in missing:
                itemList = item.splitlines()
                msg = f"{itemList[0]} ({itemList[1]})"
                print(msg)
                outStrings.append(msg)
        return outStrings

    