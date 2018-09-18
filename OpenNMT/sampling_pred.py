
import random

data = []

for l1, l2, l3 in zip( open('predfiles/test.tgt','r'), open('predfiles/new_default_e15.txt','r'), open('predfiles/local_k30_a7_e15l.txt','r')):
    data.append( [l1,l2,l3] )


fout = open('scoring.txt','w')
fout_idx = open('scoring_idx.txt','w')

random.shuffle(data)

for i,d in enumerate(data):
    l1 = d[0]
    l2 = d[1]
    l3 = d[2]

    if i > 50:
        break
    if random.random() < 0.5:
        fout.write('reference'+ '\t' + l1 + 'data1\t' +l2 + 'data2\t' + l3 + '\n')
        fout_idx .write('0'+'\n')
    else:
        fout.write('reference'+ '\t' + l1 + 'data1\t' +l3 + 'data2\t' + l2 + '\n')
        fout_idx .write('1'+'\n')







