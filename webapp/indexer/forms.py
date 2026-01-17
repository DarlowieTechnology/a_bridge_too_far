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
                                           