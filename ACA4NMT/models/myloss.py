import torch
from torch import nn
from torch.nn import functional as F


class hybrid(nn.Module):
    def __init__(self, pad, redu):
        super(hybrid, self).__init__()
        #self.cross = nn.CrossEntropyLoss(size_average = size_average, ignore_index = 0)
        self.cross = nn.CrossEntropyLoss(ignore_index=pad, reduce=redu)
        self.k = 20
        self.alpha = 0

    def duploss(self, predicted):
        summ=0
        #start = idx * predicted
        #end = idx * predicted + shard_size

        soft = F.softmax(predicted)
        _, predidx = torch.max(predicted, dim=2)

        for i_seq in range(1, predicted.size(0)):
            if i_seq > self.k:
                cal_start = i_seq - self.k
            else:
                cal_start = 0
            indices = predidx[cal_start:i_seq].transpose(1,0)

            summ += torch.gather(soft[i_seq], 1, indices).sum()

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

        totalloss = ((1 - self.alpha) * cross) + (self.alpha * local)
        #print(cross.sum(), local)
        #return totalloss
        return (cross,local)

