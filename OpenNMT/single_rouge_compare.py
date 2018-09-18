
from rouge import Rouge
import sys
import os

#fname = 'predfiles/default_e10.txt'
#if len(sys.argv) > 1:
#    fname = sys.argv[1]


def rouge_test (fname):
    fout = open('single_rouge_comp.txt','w')
    rouge = Rouge()
    hyp = []
    gold = []

    for idx,(b, h, g, s) in enumerate(zip(open('predfiles2/default_e15l.txt','r'),open('predfiles/local_k30_a7_e15l.txt','r'),open('/shared/summ_data/data300_80/test.tgt','r'), open('/shared/summ_data/data300_80/test.src','r' ))):
    #for h, g in zip(open('defaulte10.txt','r'),open('/shared/summ_data/data300_80/test.tgt','r')):
        fout.write(s+'\n')
        fout.write(g+'\n')
        scores = rouge.get_scores(b, g)
        fout.write ('baseline\n')
        fout.write(b)

        #fout.write ('rouge-1')
        scores = scores[0]
        fout.write ('rouge-1 '+ str(round(scores['rouge-1']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-1']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-1']['f'],5)) + '\n')

        #fout.write('rouge-2')
        fout.write ('rouge-2 ' + str(round(scores['rouge-2']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-2']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-2']['f'],5)) + '\n')

        #fout.write('rouge-l')
        fout.write ('rouge-l '+str(round(scores['rouge-l']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-l']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-l']['f'],5))+'\n')

        scores = rouge.get_scores(h, g)
        scores=scores[0]
        fout.write ('repeat_loss\n')
        fout.write(h)

        #fout.write ('rouge-1')
        fout.write ('rouge-1 ' + str(round(scores['rouge-1']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-1']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-1']['f'],5))+'\n')

        #fout.write('rouge-2')
        fout.write ('rouge-2 ' + str(round(scores['rouge-2']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-2']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-2']['f'],5)) + '\n')

        #fout.write('rouge-l')
        fout.write ('rouge-l ' + str(round(scores['rouge-l']['p'],5)) + ' ')
        fout.write (str(round(scores['rouge-l']['r'],5)) + ' ')
        fout.write (str(round(scores['rouge-l']['f'],5)) + '\n')

        fout.write('\n\n')

        if len(h.strip())<1:
            print('error',idx)


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

search("predfiles")
#search('dummy')
filelist.sort()
#filelist = ['predfiles/new_default_e15.txt','predfiles/local_k30_a7_e15l.txt','predfiles/local_k30_a5_e15.txt']
for i in filelist:
    #if i.find('e15')>=0:
    #    continue
    rouge_test(i)
    break
