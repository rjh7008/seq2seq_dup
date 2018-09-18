import os
import io
import sys





filelist = []

#fout = open(sys.argv[2],'w')
def len_test(fname):
    t=0
    for i,line in enumerate(open(fname,'r')):
        line=line.strip()
        l = line.split(' ')
        dic = {}
        for w in l:
            if w not in dic:
                dic[w] = 1
        uniq = len(dic)
        lens = len(l)

        t += (uniq/lens)
        

    print (t/(i+1))




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
search('predfiles')

filelist.sort()
#filelist = ['predfiles/local_tfonly95_k30_a8_e14.txt','predfiles/local_tfonly95_k30_a8_e15.txt']
for cc in filelist:
    #if cc.find('e10')>=0:
    #    continue
    print (cc.split('/')[-1])
    len_test(cc)

