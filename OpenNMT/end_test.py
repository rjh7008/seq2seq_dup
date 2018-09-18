import os
import io
import sys





filelist = []

#fout = open(sys.argv[2],'w')
def len_test(fname):
    t=0

    for i,line in enumerate(open(fname,'r')):
        line=line.strip()
        if line[-1] != '.':
            t+=1
            #print(line)
            #print (line)
        #t += l

    print (t)




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
search('komodels/predfiles')

filelist.sort()
#filelist = ['predfiles2/default_e15l.txt']
#filelist = ['predfiles/default_e15l.txt', 'predfiles/local_k30_a7_e15l.txt']
for cc in filelist:
    #if cc.find('e10')>=0:
    #    continue
    print (cc.split('/')[-1])
    len_test(cc)

