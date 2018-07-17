import torch
import torch.utils.data as data
from torch.autograd import Variable

from collections import Counter

import os

class summ(data.Dataset):
  def __init__(self, path='', src='', tar='', opt=None, vocab = None):
    self.src = []
    self.tar = []

    if vocab is None:
      self.vocab = self.build_vocab(path, src, tar, opt.max_vocab)
    else:
      self.vocab = vocab

    print ('vocab word : ',len(self.vocab['i2w']))
    for i,(s,t) in enumerate(zip( open(os.path.join(path,src),'r'),open(os.path.join(path,tar),'r') )):    
      tmp=[]
      s=s.strip()
      t=t.strip()
      for w in s.split(' '):
        if w in self.vocab['w2i']:
          tmp.append(self.vocab['w2i'][w])
        else:
          tmp.append(self.vocab['w2i']['<unk>'])

      self.src.append(tmp)

      tmp=[]
      for w in t.split(' '):
        if w in self.vocab['w2i']:
          tmp.append(self.vocab['w2i'][w])
        else:
          tmp.append(self.vocab['w2i']['<unk>'])

      self.tar.append(tmp)

  def __getitem__(self,idx):
    return self.src[idx],self.tar[idx]

  def __len__(self):
    return len(self.src)

  def build_vocab(self, path, src, tar, vs):
    
    word2idx = { '<pad>' : 0, '<s>' : 1, '</s>' : 2, '<unk>' : 3 }
    idx2word = ['<pad>', '<s>', '</s>', '<unk>']
    vocab_counter = Counter()

    wi = 4
    for i,(s,t) in enumerate(zip( open(os.path.join(path,src),'r'), open(os.path.join(path,tar),'r') )):
      s=s.strip()
      t=t.strip()
      for w in s.split(' '):
        vocab_counter[w] += 1
      for w in t.split(' '):
        vocab_counter[w] += 1
    print ('total word : ', len(vocab_counter))
    if len( vocab_counter.most_common() ) > vs - 4:
      vocab_cnt = vocab_counter.most_common( vs - 4 )
    else:
      vocab_cnt = vocab_counter

    for i,word in enumerate(vocab_cnt):
      idx2word.append(word[0])
      word2idx[word[0]] = i+4

    vocab = {'w2i' : word2idx, 'i2w' : idx2word}

    torch.save(vocab,'summ.vocab')
    
    return vocab

def pad_collate_fn(batch):
    srcMax = 0
    tarMax = 0

    u = []
    y = []
    for s, t in batch:
        if srcMax < len(s):
            srcMax = len(s)
        if tarMax < len(t):
            tarMax  = len(t)

    for i in range(len(batch)):
        batch[i][0].extend([0 for _ in range(srcMax - len(batch[i][0]))])
        batch[i][1].extend([0 for _ in range(tarMax - len(batch[i][1]))])
        u.append(batch[i][0])
        y.append(batch[i][1])

    return Variable(torch.LongTensor(u)).cuda(), Variable(torch.LongTensor(y)).cuda()

'''
class summ(data.Dataset):
    filename=
    dirname=

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
'''

