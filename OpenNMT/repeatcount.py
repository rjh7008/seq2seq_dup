import os
import io
import sys




#fout = open(sys.argv[2],'w')

freq = [0,0,0,0,0,0]
for i,line in enumerate(open(sys.argv[1],'r')):
    line=line.strip()

    words = line.split(' ')

    worddic = {}

    for w in words:
        if w in worddic:
            if worddic[w] < 5:
                worddic[w] += 1
        else:
            worddic[w] = 1

    for k in worddic:
        freq[worddic[k]] += 1



print (freq)

for f in freq[2:]:
    print ( round(f/(i+1),2) , end = ' ')
print()

