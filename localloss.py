import torch
from torch import nn
from torch.nn import functional as F


class hybrid(nn.Module):
  def __init__(self):
    super(hybrid,self).__init__()
    self.crossloss=nn.CrossEntropyLoss(ignore_index = 0)
    
    
  def duploss(self,pred,history, num):
    soft = F.softmax(pred,dim=1)
    summ = 0
    
    #print (history)
    #print(history.size())
    #print(soft.size())
    #print(num)


    for i in range(num):#history (b,s) pred (b,v)  # seq
      for j in range(soft.size(0)):# batch
        #print (i, int(history[j][i].item()))
        summ += soft[j][int(history[i][j].item())]
        #pred[:,history[:,i]]

    return summ
    
  def forward(self, pred, y, history, num):
    loss = 0

    cross = self.crossloss(pred,y)
    local = self.duploss(pred,history, num)

    loss = cross + local

    return loss


