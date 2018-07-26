
import torch
from torch import nn
from torch.nn import functional as F


class hybrid(nn.Module):
    def __init__(self, weight, size_average):
        super(hybrid, self).__init__()
        self.cross = nn.CrossEntropyLoss(weight, size_average = size_average, ignore_index = 0)
        self.k = 30
        self.alpha = 0.5

    def duploss(self, predicted):
        summ=0
        #start = idx * predicted
        #end = idx * predicted + shard_size

        soft = F.softmax(predicted, dim=-1)
        _, predidx = torch.max(soft, dim=2)
        
        batch_size = predicted.size(1)

        for i_seq in range(1, predicted.size(0)):
            if i_seq > self.k:
                cal_start = i_seq - self.k
            else:
                cal_start = 0

            indices = predidx[cal_start:i_seq].transpose(1,0)
            zerovec = torch.zeros(indices.size()).cuda()

            #summ += torch.gather(soft[i_seq], 1, indices).sum()
            
            #print (_[i_seq].size())
            #print (predidx[i_seq])
            #print (indices)


            summ += torch.where(predidx[i_seq].view(batch_size , 1).expand(batch_size, indices.size(1)) == indices, _[i_seq].view(batch_size, 1).expand(batch_size, indices.size(1)), zerovec).sum()

            #exit(1)

            #for i_batch in range(predicted.size(1)): # batch_size
            #    indices = predidx[cal_start:i_seq, i_batch]
            #    tsumm += predicted[i_seq, i_batch, indices].sum()
            #print(summ)
            #print (tsumm)
            #exit(1)

        return summ

    def forward(self, scores, gtruth, pred):
        #predicted = softmax
        #predidx = maxidx

        totalloss = 0

        cross = 0
        local = 0

        if self.alpha == 0:
            cross = self.cross(scores, gtruth)
        else:
            cross = self.cross(scores, gtruth)
            local = self.duploss(pred)
        #print (cross.item(), local.item())

        if local.item() != 0:
            totalloss = ((1 - self.alpha) * cross) + (self.alpha * local)
        else:
            totalloss += ((1-self.alpha) * cross)
        
        return totalloss
