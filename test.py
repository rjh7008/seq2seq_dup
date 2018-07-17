import torch
from torch import nn
from torch.autograd import Variable

#batch = 4

a = Variable(torch.randn(4,10))#pred
b = Variable(torch.randint( low=1,high=5, size=(4,5), dtype=torch.long) )#history

#for i in a:
#  for j in i:
#    print(j.item(), end=' ')
#  print()
print (a)
print (b)
print (a.size(),b.size())

c = torch.nn.functional.softmax(a,dim=1)
print (c)

for i in range(b.size(1)):
  print(b[:,i])
  for j in range(b.size(0)):
    print (c[j][b[j][i]], end=' ')
  print()
  #print(a[:,0])



'''
print (torch.__version__)
a = torch.FloatTensor([0,1]).view(-1,1,1)

b = nn.GRU(1,1,bidirectional=True,bias=False)
for k, v  in b.named_parameters():
   setattr(b, k, torch.nn.Parameter(torch.ones_like(v.data)))

print (a)
print (a.size())
print (b)
c,_= b(a)
print (c[0])
print (c[1])

print (_)
print (c.size())
print (_.size())

print('==')

a = torch.FloatTensor([1,0]).view(-1,1,1)

print (a)
print (a.size())
print (b)
c,_= b(a)
print (c[0])
print (_)
print (c.size())
print (_.size())
'''
