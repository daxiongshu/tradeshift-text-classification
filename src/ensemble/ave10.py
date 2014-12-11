import sys
subdir=sys.argv[1]
f1=open(subdir+'ave3-tk16-19-20x.csv')
f2=open(subdir+'submission-nn2-1r.csv')
f3=open(subdir+'submissiontk28-33.csv')
f3.readline()
#f2=open('../idiot/kaggle-2014-criteo-master/submission.csv')

fo=open(subdir+'nn2-tk28-ave3-tk16-19-20.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()

c=0
for l in f1:
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
 #   p3=float(n.split(',')[-1])
    p=p1#*0.5+p2*0.2+p3*0.3
    if c%33==32:
        p2=float(f2.readline().split(',')[-1])
        p3=float(f3.readline().split(',')[-1])
        p=(p2+p1+p3)/3
    fo=open(subdir+'nn2-tk28-ave3-tk16-19-20.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()
fo.close()
f3.close()
