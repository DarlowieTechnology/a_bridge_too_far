from django import forms

import sys

# local
sys.path.append("..")
sys.path.append("../..")

from testQueries import TESTSET


class QueryForm(forms.Form):
    query = forms.CharField(label="Query:", max_length=100)

class SettingsColumnOne(forms.Form):
    cutIssueDistance = forms.FloatField(label="Vector cut-off distance: ", max_value=1.0, min_value=0.0)
    QUERYTYPESORIGINAL = forms.BooleanField(label="Original vector query: ", required=False)
    QUERYTYPESHYDE = forms.BooleanField(label="HyDE vector query: ", required=False)
    QUERYTYPESMULTI = forms.BooleanField(label="Multi vector query: ", required=False)
    QUERYTYPESREWRITE = forms.BooleanField(label="Rewrite vector query: ", required=False)
    semanticRetrieveNum = forms.IntegerField(label="Maximum vector records: ", max_value=1000, min_value=1)

class SettingsColumnTwo(forms.Form):
    bm25sCutOffScore = forms.FloatField(label="BM25s cut-off score: ", max_value=1.0, min_value=0.0)
    QUERYTYPESBM25SORIG = forms.BooleanField(label="Original BM25s query: ", required=False)
    QUERYTYPESBM25PREP = forms.BooleanField(label="Prepared BM25s query: ", required=False)
    bm25sRetrieveNum = forms.IntegerField(label="Maximum BM25s records: ", max_value=1000, min_value=1)
    TOKENIZERTYPESSTOPWORDSEN = forms.BooleanField(label="BM25s stop words: ", required=False)
    TOKENIZERTYPESSTEMMER = forms.BooleanField(label="BM25s stemmer: ", required=False)

class SettingsColumnThree(forms.Form):
    queryPreprocess = forms.BooleanField(label="Process query after transform: ", required=False)
    queryCompress = forms.BooleanField(label="Compress query after transform: ", required=False)
    rrfTopResults = forms.IntegerField(label="Maximum number of RRF results: ", max_value=1000, min_value=1)
    wellknownTestSet = forms.ChoiceField(label="Well-known test set: ", required=False, choices=[(TESTSET.NOTEST, 'None'), (TESTSET.XSS, 'XSS issues'), (TESTSET.CREDS, 'Credentials issues')])
