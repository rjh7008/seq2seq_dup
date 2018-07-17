import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils import data

import argparse

import opt
import mydataset
import dataloader
import model

import torchtext

parser = argparse.ArgumentParser(description='main.py')
opt.model_opts(parser)
opts = parser.parse_args()

'''
def mydataloader(sf, tf):
  traindata = mydataset.summ(sf,tf,src='article_src.txt',tar='article_tar.txt',path='../../ACA4NMT/kodata/')
  #devdata = mydatasets.summ(sf,tf,filename='article_tar.txt',path='../../ACA4NMT/kodata/')
  sf.build_vocab(max_size=40000)
  #tf.build_vocab(max_size=40000)
  train_iter = data.BucketIterator.splits( (train_data),
                                           batch_sizes=(opts.batch_size) )
  return train_iter


src_field = data.Field()
tgt_field = src_field
#tgt_field = data.Field()

train_iter,dev_iter = mydataloader(src_field, tgt_field)
'''

v = torch.load('summ.vocab')

# dl = dataloader.summ(path='../../ACA4NMT/kodata/', src='article_src.txt', tar='article_tar.txt', opt=opts )
dl = dataloader.summ(path='../', src='dev.src', tar='dev.tgt', opt=opts, vocab=v )
dataloader = data.DataLoader(dataset=dl, batch_size=opts.batch_size, collate_fn=dataloader.pad_collate_fn)

model = model.s2s(opts).cuda()
model.load_state_dict(torch.load('local_4.model'))
#model.load_state_dict(torch.load('s2s_6.model'))
model.eval()
for i,(s,t) in enumerate(dataloader):
    batch_loss,ploss,result = model(s,t)
    #print(result)
    print(result.size())
    result = result.view(-1)
    print(result.size())
    for i in result:
      print(dl.vocab['i2w'][int(i.item())],end=' ')
    print()
    t=t.view(-1)
    for i in t:
      print(dl.vocab['i2w'][int(i.item())],end=' ')
    print()
    break

