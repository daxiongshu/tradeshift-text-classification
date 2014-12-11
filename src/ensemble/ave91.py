import sys
subdir=sys.argv[1]
f1=open(subdir+'xgb-raw-d20-e0.07-min6-tree150.csv')
f2=open(subdir+'xgb-y33-d100-e0.1-min7-tree150.csv')

fo=open(subdir+'xgb-raw','a')
fo.write(f1.readline())
fo.close()
f2.readline()

c=0
for l in f1:
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p=p1
    if c%33==32:
        p2=float(f2.readline().split(',')[-1])
        p=p1*0.5+p2*0.5
    fo=open(subdir+'xgb-raw','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()

fo.close()
