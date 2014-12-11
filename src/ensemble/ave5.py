import sys
subdir=sys.argv[1]
f=open(subdir+'ave3.csv')
f2=open(subdir+'submissiontk16.csv')
f3=open(subdir+'submissiontk19.csv')
f4=open(subdir+'submissiontk20.csv')


fo=open(subdir+'ave3-tk16-19-20x.csv','w')
fo.write(f.readline())
f2.readline()
f3.readline()
f4.readline()


c=0
for l,m,n,o in zip(f,f2,f3,f4):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    p3=float(n.split(',')[-1])
    p4=float(o.split(',')[-1])
    p=((p4+p2+p3)/3+p1)/2
    
    fo.write(idx+','+str(p)+'\n')
    c+=1

fo.close()
f.close()
f2.close()
f3.close()
f4.close()
