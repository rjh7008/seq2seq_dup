import os
import io
import sys





filelist = []

#fout = open(sys.argv[2],'w')
def freq_test(fname, gram):
    freq = [0,0,0,0,0,0]
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
        
        for w in words:
            if w in worddic:
                if worddic[w] < 5:
                    worddic[w] += 1
            else:
                worddic[w] = 1
        f = False
        for k in worddic:
            freq[worddic[k]] += 1
            if not f and worddic[k] >=5:
                #print (line)
                f=True
    #print (fname)
    #print (freq)
    try:
        for f in freq[2:]:
            #print (f, end = ' ')
            print ( round(f/(i+1),2) , end = ' ')
            
            #print ( f , end = ' ')
        print()
    except:
        return
    #print()
    #print()


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
#filelist = ['predfiles/new_default_e15.txt','predfiles/local_k30_a7_e15l.txt','predfiles/local_k30_a5_e15.txt']
#filelist = ['predfiles2/default_e15l.txt']
for cc in filelist:
    #if cc.find('e15')>=0:
    #    continue
    print (cc.split('/')[-1])
    freq_test(cc,1)
    freq_test(cc,2)
    freq_test(cc,3)
    freq_test(cc,4)
    freq_test(cc,5)

