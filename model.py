import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import random
import localloss


class encoder(nn.Module):
  def __init__(self, embedding_size, hidden_size, vocab_size, embedding_layer, num_layer):
    super(encoder,self).__init__()
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_vocab = vocab_size
    self.num_layer = num_layer
    
    self.embed = embedding_layer
    self.rnn = nn.GRU( self.embedding_size, self.hidden_size, num_layers = self.num_layer, bidirectional = True )

  def forward(self, x, h):
    x = x.transpose(1,0)

    embeded = self.embed(x)

    o,h = self.rnn(embeded,h)
    
    return o,h
  def init_hidden(self,seq_len, batch_size, hidden_size):
    return Variable(torch.randn(self.num_layer * 2, batch_size, hidden_size)).cuda()
    

class Atten(nn.Module):
  def __init__(self, hidden_size):
    super(Atten, self).__init__()
    self.hidden_size = hidden_size
    self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)

  def forward(self, enc, dec):
    e = Variable(torch.zeros(enc.size(0),enc.size(1) )).cuda()
    for ei in range(enc.size(0)):
      t = self.attn(enc[ei])
      t = t.unsqueeze(0).transpose(0,1).transpose(1,2)#.transpose(0,1)
      #print ('ssssssss',dec.transpose(1,0).bmm(t).squeeze().size())
      e[ei] = dec.transpose(1,0).bmm(t).squeeze(2).squeeze(1)
    e=e.transpose(1,0)
    a = F.softmax(e,dim=1).unsqueeze(2)

    out = enc.transpose(1,0).transpose(1,2).bmm(a)
    out = out.squeeze(2)
    return a, out

class decoder(nn.Module):
  def __init__(self, hidden_size, vocab_size, embedding_size, embedding_layer):
    super(decoder,self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.embed = embedding_layer

    self.rnn = nn.GRU(self.hidden_size*2 + self.embedding_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.vocab_size)
    
  def forward(self,input, att, h):
    embeded = self.embed(input)
    rnn_input = torch.cat((embeded, att.unsqueeze(0)),dim=2)
    out, h = self.rnn(rnn_input, h)
    out = self.out(out).squeeze(0)

    return out,h

class s2s(nn.Module):
  def __init__(self,args):
    super(s2s,self).__init__()
    self.args = args

    self.shared_embed = nn.Embedding(args.max_vocab, args.embed_size, padding_idx=0)
    self.enc = encoder(args.embed_size, args.hidden_size, args.max_vocab, self.shared_embed, args.layers)
    self.atten = Atten(args.hidden_size)
    self.dec = decoder(args.hidden_size, args.max_vocab, args.embed_size, self.shared_embed)

    self.tf_ratio = 0.5

  def forward(self,x,t):
    #print (t.size())
    #criterion = nn.CrossEntropyLoss()
    criterion = localloss.hybrid()

    h = self.enc.init_hidden(x.size(1),x.size(0),self.args.hidden_size)
    encoder_out,h = self.enc(x,h)

    decoder_out = Variable(torch.randn(1,self.args.batch_size, self.args.hidden_size)).cuda()
    lastword = Variable(torch.ones( (1, x.size(0)),dtype=torch.long )).cuda()
    dh = h[0:1,:] + h[1:2,:]

    loss =0
    tf = True if random.random() < self.tf_ratio else False

    lastwords = Variable(torch.zeros( t.size(1), t.size(0) ))
    tf = False

    if tf:
      for di in range(t.size(1)):# batch first
        attn, attn_out = self.atten(encoder_out, decoder_out )
        dout,dh = self.dec(lastword, attn_out, dh)

        loss += criterion( dout, t[:,di], lastwords, di)

        lastword = t[:,di].view(1,-1)

        _,lastwords[di] = torch.max(dout, dim = 1)
        

        #_,lastword =torch.max(dout,dim=1)
        #break
    else:
      for di in range(t.size(1)):
        attn, attn_out = self.atten(encoder_out, decoder_out )
        dout,dh = self.dec(lastword, attn_out, dh)

        loss += criterion( dout, t[:,di], lastwords, di)

        #lastword = t[:,di].view(1,-1)
        _,lastword =torch.max(dout,dim=1)
        lastword = lastword.view(1,-1)
        lastwords[di] = lastword
        #break
    
    return loss, loss.item()/t.size(1), lastwords

'''
def model_opts(parser):
parser.add_argument('-batch_size', default=16, type=int,
parser.add_argument('-hidden_size', default=256, type=int,
parser.add_argument('-embed_size', type=int, default=200,
parser.add_argument('-max_vocab', default=40000, type=int,
parser.add_argument('-mode', default='train', type=str,
parser.add_argument('-layers', default=1, type=int,
'''

