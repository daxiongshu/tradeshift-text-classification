import sys
subdir=sys.argv[1]
f1=open(subdir+'xgb-part1-d18-e0.09-min6-tree120-xgb_base.csv')
f2=open(subdir+'xgb-part1-d20-e0.07-min6-tree150-xgb_base.csv')
f3=open(subdir+'xgb-y33-d30-e0.1-min6-tree80-all-sparse.csv')
fo=open(subdir+'xgb-base-xgb-meta-sparse.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()
f3.readline()
c=0
for l,m in zip(f1,f2):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    
    p=p1*0.5+p2*0.5
    if c%33==32:
        p3=float(f3.readline().split(',')[-1])
        p=(p1+p2)/2*0.7+p3*0.3
    fo=open(subdir+'xgb-base-xgb-meta-sparse.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()

fo.close()
