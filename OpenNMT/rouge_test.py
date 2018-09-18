
from rouge import Rouge
import sys
import os

#fname = 'predfiles/default_e10.txt'
#if len(sys.argv) > 1:
#    fname = sys.argv[1]


def rouge_test (fname):

    rouge = Rouge()
    hyp = []
    gold = []

    for idx,(h, g) in enumerate(zip(open(fname,'r'), open('/shared/summ_data/data300_80/test.tgt','r'))): #open('/shared/summ_data/data300_80/small_test.tgt','r'))):
    #for h, g in zip(open('defaulte10.txt','r'),open('/shared/summ_data/data300_80/test.tgt','r')):
        if len(h.strip())<1:
            print('error',idx)
        hyp.append(h.strip())
        gold.append(g.strip())

    #print (len(hyp))
    #print (len(gold))
    try:
        scores = rouge.get_scores(hyp, gold,avg=True)
        #print(scores)
        print ('',fname.split('/')[-1])
        print (' p r f')

        #print ('rouge-1')
        print ('rouge-1',round(scores['rouge-1']['p'],5), end = ' ')
        print (round(scores['rouge-1']['r'],5), end = ' ')
        print (round(scores['rouge-1']['f'],5))
    
        #print('rouge-2')
        print ('rouge-2',round(scores['rouge-2']['p'],5), end = ' ')
        print (round(scores['rouge-2']['r'],5), end = ' ')
        print (round(scores['rouge-2']['f'],5))
    
        #print('rouge-l')
        print ('rouge-l',round(scores['rouge-l']['p'],5),end = ' ')
        print (round(scores['rouge-l']['r'],5),end = ' ')
        print (round(scores['rouge-l']['f'],5))
    except:
        return

filelist = []

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                filelist.append(full_filename)
    except PermissionError:
        pass

search("komodels/predfiles")
#search('dummy')
filelist.sort()
#filelist = ['predfiles/new_default_e15.txt','predfiles/local_k30_a7_e15l.txt','predfiles/local_k30_a5_e15.txt']
for i in filelist:
    #if i.find('e15')>=0:
    #    continue
    rouge_test(i)

