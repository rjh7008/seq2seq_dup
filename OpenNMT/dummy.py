
import torch


arr = [[0.0]]


for ts in range(90):
    t=[0]
    for s in range(ts+1):
        a = 1

        t.append( pow(0.95,s)  )
    t = torch.FloatTensor(t)

    arr.append(t)
    print (t)
    if ts>10:
        break
    

print()
#print (arr)


print (arr[3])

#arr = torch.FloatTensor(arr)

b= arr[3].view(1,-1)

print (b)
print (b.size())

print ('repeat')
print (b.repeat(5,1))




