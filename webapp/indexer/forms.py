from django import forms

import sys

# local
sys.path.append("..")
sys.path.append("../..")

class IndexerForm(forms.Form):

    def __init__(self, *args, **kwargs):
            fileList = kwargs.pop('fileList')
            super(IndexerForm, self).__init__(*args, **kwargs)

            choices = []
            for i, fileName in enumerate(fileList):
                myTuple = (fileName, fileName)
                choices.append(myTuple)

            self.fields['File Choices'] = forms.ChoiceField(choices=choices, required=False,
                widget=forms.RadioSelect(attrs={
                    'onchange': f'handleSelection()',  # JS event handler
                })                                                       
            )
            
            self.fields['inputFile'] = forms.CharField(label="File:", max_length=1000, required=False,
                    widget=forms.TextInput(attrs={
                        'size': '200'
                    })
            )


class SettingsColumnOne(forms.Form):
    LoadDocument = forms.BooleanField(label="Load document", required=False)
    stripWhiteSpace = forms.BooleanField(label="Strip whitespace", required=False)
    convertToLower = forms.BooleanField(label="Convert to lower case", required=False)
    convertToASCII = forms.BooleanField(label="Convert UTF to ASCII", required=False)
    singleSpaces = forms.BooleanField(label="Replace separators by single spaces", required=False)


class SettingsColumnTwo(forms.Form):
    rawTextFromDocument = forms.BooleanField(label="Extract raw records", required=False)
    finalJSONfromRaw = forms.BooleanField(label="Create final JSON", required=False)
    prepareBM25corpus = forms.BooleanField(label="Prepare BM25 corpus", required=False)
    completeBM25database = forms.BooleanField(label="Complete BM25 database", required=False)


class SettingsColumnThree(forms.Form):
    vectorizeFinalJSON = forms.BooleanField(label="Create vector database", required=False)
    DISPLAYjira_export = forms.BooleanField(label="Create Jira database", required=False)
