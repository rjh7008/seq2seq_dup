import os
import io
import sys





filelist = []
def len_test(l):
    t=0
    m=0
    try:
        for i,line in enumerate(l):
            if len(line)<1:
                m+=1
                continue
            dic = {}
            for w in line:
                if w not in dic:
                    dic[w] = 1
            uniq = len(dic)
            lens = len(line)

            t += (uniq/lens)

        print ( 1- (t/(i+1-m)))
    except:
        print(i+1)
        print(line)
        exit(1)

#fout = open(sys.argv[2],'w')
def freq_test(fname, gram):
    freq = [0,0,0,0,0,0]
    sents = []
    for i,line in enumerate(open(fname,'r')):
        line=line.strip()

        words = line.split(' ')
        tmp = line.split(' ')
        words = []

        for j in range(len(tmp)-gram+1):
            t = ''
            for k in range(gram):
                t+= tmp[ j + k ] + ' '
            t=t.strip()
            words.append(t)
        worddic = {}
        sents.append(words)
    len_test(sents)

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                filelist.append(full_filename)
                #freq_test(full_filename)

    except PermissionError:
        pass

#search("../deep_summ/predfiles")
#search('predfiles')
search('komodels/predfiles')
filelist.sort()
#filelist = ['predfiles/new_default_e15.txt','predfiles/local_k30_a7_e15l.txt','predfiles/local_k30_a5_e15.txt']

for cc in filelist:
    #if cc.find('e15')>=0:
    #    continue
    print (cc.split('/')[-1])
    freq_test(cc,1)
    freq_test(cc,2)
    freq_test(cc,3)
    freq_test(cc,4)
    #freq_test(cc,5)







