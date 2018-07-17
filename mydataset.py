import torch
import re
import os
import random
import tarfile
import urllib
from torchtext import data



class summ(data.Dataset):
    filename=''
    dirname=''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    def __init__(self,text_field, label_field, src=None, tar=None, path=None, examples=None, shuffle=False, **kwargs):
        #text_field.preprocessing = data.Pipeline(clean_str)

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []
            #with open(os.path.join(path, filename), errors='ignore') as f:
            for i,(s,t) in enumerate(zip( open(os.path.join(path,src),'r'),open(os.path.join(path,tar),'r') )):
                examples+= [data.Example.fromlist( [ s, t ], fields)]
                #examples += [
                #    data.Example.fromlist([line, 'negative'], fields) for line in f]
        if shuffle:
            random.shuffle(examples)
        super(nsmc, self).__init__(examples, fields, **kwargs)


