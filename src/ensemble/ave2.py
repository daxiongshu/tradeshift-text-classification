import sys
subdir=sys.argv[1]
f1=open(subdir+'submissiontk7.csv')
f2=open(subdir+'submission-myn2.csv')
f3=open(subdir+'submissiontk9.csv')
fo=open(subdir+'ave2.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()
f3.readline()
c=0
for l,m,n in zip(f1,f2,f3):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    p3=float(n.split(',')[-1])
    p=p1+p2+p3
    fo=open(subdir+'ave2.csv','a')
    fo.write(idx+','+str(p/3)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()
f3.close()
fo.close()
