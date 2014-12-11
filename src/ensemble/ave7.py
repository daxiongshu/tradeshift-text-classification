import sys
subdir=sys.argv[1]

f=open(subdir+'tk33-ave3-tk16-19-20-y33.csv')
f2=open(subdir+'submissiontk36x.csv')
fo=open(subdir+'36x-tk33-ave.csv','w')
fo.write(f.readline())
f2.readline()
for l,m in zip(f,f2):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    fo.write(idx+','+str((p1+p2)/2)+'\n')
fo.close()
f.close()
f2.close()
