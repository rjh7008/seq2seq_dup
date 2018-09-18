
import torch
from torch import nn
from torch.nn import functional as F




class hybrid(nn.Module):
    def prepare_alphas(self):
        arr = [[0.0]]


        for ts in range(90):
            t=[]
            for s in range(ts+1):
                a = 1
                #t.append( 0.5)
                t.append( pow(0.95,s)  )
            t = torch.cuda.FloatTensor(t)

            arr.append(t)
        return arr


    def __init__(self, weight, size_average):
        super(hybrid, self).__init__()
        self.cross = nn.CrossEntropyLoss(weight, size_average = size_average, ignore_index = 0)
        self.k = 30
        #self.alpha = 0.7
        self.alpha=0.7
        
        self.seq_alpha = self.prepare_alphas()
        #print (self.seq_alpha)

        print ('loss k :', self.k)
        print ('loss alpha :',self.alpha)
    def calc_uniq(self, predidx):
        tu=0
        for i_batch in range(predidx.size(1)):
            u=0
            dic = []
            wordcnt = 0
            for i_s in range(predidx.size(0)):
                idx = predidx[i_s,i_batch].item()
                if idx == 0:
                    continue
                if idx not in dic:
                    dic.append(idx)
                wordcnt += 1
            tu += len(dic) / wordcnt
            #print (len(dic) / wordcnt)
        #print ('uniq')
        tu /= predidx.size(1)
        return tu
    def duploss(self, predicted):
        summ=0
        s2=0
        #start = idx * predicted
        #end = idx * predicted + shard_size

        soft = F.softmax(predicted, dim=-1)
        #soft = predicted
        _, predidx = torch.max(soft, dim=2)
        #print (predidx.size())
        #print ('data')
        #print (predidx.data)
        #for zz in predidx.transpose(1,0):
        #    for zzz in zz:
        #        print (zzz.item(), end=' ')
        #    print ()
        #print ('uniq')
        tu=0
        for i_batch in range(predidx.size(1)):
            u=0
            dic = []
            wordcnt = 0
            for i_s in range(predidx.size(0)):
                idx = predidx[i_s,i_batch].item()
                if idx == 0:
                    continue
                if idx not in dic:
                    dic.append(idx)
                wordcnt += 1
            tu += len(dic) / wordcnt
            #print (len(dic) / wordcnt)
        #print ('uniq')
        tu /= predidx.size(1)


        for i_seq in range(1, predicted.size(0)):
            if i_seq > self.k:
                cal_start = i_seq - self.k
            else:
                cal_start = 0
            
            '''
            indices = predidx[cal_start:i_seq].transpose(1,0)
            tmp = torch.gather(soft[i_seq], 1, indices)
            for i_batch in range(predicted.size(1)):
                if predidx[i_seq,i_batch].item() == 0:
                    continue
                t = torch.unique(tmp[i_batch].cpu()).cuda()
                print (t)
                summ += t.sum()
            '''
            #exit(1)
            #summ += tmp.sum()
            #print (soft[i_seq].size(), indices.size())
            
            indices = predidx[cal_start:i_seq].transpose(1,0)
            t = torch.gather(soft[i_seq], 1, indices)

            bbatch_size = t.size(0)
            sseq_size = t.size(1)

            #print(t.size())
            
            alp = self.seq_alpha[sseq_size].view(1,-1).repeat(bbatch_size,1)
            #print ('alp')
            #print (alp)
            #print ('t')

            #print (t)
            #print ('t*alp')

            #print (t * alp)

            #exit(1)
            t=t.sum()

            if i_seq> self.k:
                summ += (t / (self.k))
            else:
                summ += (t / (i_seq))

            #print (indices)
            #print (indices.size())
            #exit(1)
            '''
            t=0
            for i_batch in range(predicted.size(1)):
                if predidx[i_seq,i_batch].item() == 0:
                    continue
                indices = predidx[cal_start:i_seq,i_batch]
                indices = torch.unique(indices.cpu()).cuda()
                t += torch.gather(soft[i_seq,i_batch],0,indices).sum()
                summ += t
            if i_seq>self.k:
                summ+=(t/(self.k))
            else:
                summ += (t/(i_seq))

            '''

            #print(summ, s2)
            #exit(1)

            #for i_batch in range(predicted.size(1)): # batch_size
            #    indices = predidx[cal_start:i_seq, i_batch]
            #    tsumm += predicted[i_seq, i_batch, indices].sum()
            #print(summ)
            #print (tsumm)
            #exit(1)

        return summ, tu

    def forward(self, scores, gtruth, pred):
        #predicted = softmax
        #predidx = maxidx

        totalloss = 0

        cross = 0
        local = 0


        if self.alpha == 0:
            cross = self.cross(scores, gtruth)
            totalloss = cross
            local, dup_percent = self.duploss(pred)
        else:
            #print (pred.size())
            #print (scores.size(),gtruth.size())
            cross = self.cross(scores, gtruth)
            #print('total',cross)
            #eachloss = 0
            #for i in range(scores.size(0)):
            #    eachloss += self.cross(scores[i].view(1,-1),gtruth[i].view(1))
            #print (eachloss)
            #print (eachloss.item()/scores.size(0))
            #print('===========')

            local, dup_percent = self.duploss(pred)
            #print (cross.item(), local.item())
            totalloss = ((1 - self.alpha) * cross) + (self.alpha * local)
            #print (local)
            #exit(1)
        return totalloss,cross,dup_percent
