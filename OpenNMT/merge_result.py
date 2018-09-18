import random

fout = open('survey.txt','w')
d=[]

for i,(src,line1,line2,line3) in enumerate(zip(open('/shared/summ_data/data300_80/test.src','r'),open( 'predfiles/test.tgt','r'), open( 'predfiles2/default_e15l.txt','r'), open( 'predfiles/local_k30_a7_e15l.txt','r'))):

    d.append((src,line1,line2,line3))



random.shuffle(d)

for i,(src,line1,line2,line3) in enumerate(d):

    fout.write('원문\n' + src + '======'+'\n')
    fout.write('샘플요약\n' + line1+'\n')

    fout.write('model1 output\n')
    fout.write(line2)
    fout.write('model2 output\n')
    fout.write(line3)
    fout.write('\n\n')

#    if i == 99:
#        break

#    #fout.write( line1 + line2 + line3 + '\n')
print(i)





