
from rouge import Rouge
import sys
import os
import nltk

#fname = 'predfiles/default_e10.txt'
#if len(sys.argv) > 1:
#    fname = sys.argv[1]

#nltk.translate.bleu_score.sentence_bleu

def rouge_test (fname):

    rouge = Rouge()
    hyp = []
    gold = []
    scores=0
    for idx,(h, g) in enumerate(zip(open(fname,'r'),open('/shared/summ_data/data300_80/test.tgt','r'))):
    #for h, g in zip(open('defaulte10.txt','r'),open('/shared/summ_data/data300_80/test.tgt','r')):
        if len(h.strip())<1:
            print('error',idx)
        hyp.append(h.strip().split(' '))
        gold.append(g.strip().split(' '))
        #scores += nltk.translate.bleu_score.sentence_bleu( [g.strip().split(' ')], [h.strip().split(' ')])

    #print (len(hyp))
    #print (len(gold))
    scores += nltk.translate.bleu_score.corpus_bleu(gold, hyp)
    #scores = scores / (idx+1)
    #scores = nltk.translate.bleu_score.sentence_bleu(gold, hyp)
    print ('',fname.split('/')[-1])
    print (scores)


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

#search("../deep_summ/predfiles")
search('predfiles')
filelist.sort()
#filelist = ['predfiles/local_tfonly95_k30_a8_e14.txt','predfiles/local_tfonly95_k30_a8_e15.txt']
for i in filelist:
    #if i.find('e15')>=0:
    #    continue
    rouge_test(i)

