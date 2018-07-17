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

optim = torch.optim.Adam(model.parameters())
for epoch in range(10):
    totalloss = 0
    for i,(s,t) in enumerate(dataloader):
        #print(s.size(),t.size())
        #s = Variable(torch.randint(low=1,high=1000,size=(16,1000),dtype=torch.long)).cuda()
        #t = Variable(torch.randint(low=1,high=1000,size=(16,300),dtype=torch.long)).cuda()
        optim.zero_grad()
        batch_loss,ploss = model(s,t)
        totalloss += ploss
        if i%200 == 0 :
            print (ploss)

        batch_loss.backward()
        optim.step()
        #print (out.size())
    print (totalloss)
    torch.save(model.state_dict(),'local_'+str(epoch)+'.model')

