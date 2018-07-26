
from rouge import Rouge
import sys

rouge = Rouge()


hyp = []
gold = []

fname = 'predfiles/default_e10.txt'
if len(sys.argv) > 1:
    fname = sys.argv[1]

for h, g in zip(open(fname,'r'),open('/shared/summ_data/data300_80/test.tgt','r')):
#for h, g in zip(open('defaulte10.txt','r'),open('/shared/summ_data/data300_80/test.tgt','r')):
    hyp.append(h.strip())
    gold.append(g.strip())

scores = rouge.get_scores(hyp, gold,avg=True)


#print(scores)
print ('rouge-1')
print (scores['rouge-1']['p'])
print (scores['rouge-1']['r'])
print (scores['rouge-1']['f'])

print('rouge-2')
print (scores['rouge-2']['p'])
print (scores['rouge-2']['r'])
print (scores['rouge-2']['f'])

print('rouge-l')
print (scores['rouge-l']['p'])
print (scores['rouge-l']['r'])
print (scores['rouge-l']['f'])






