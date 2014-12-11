import sys
subdir=sys.argv[1]
f1=open(subdir+'best_online_ensemble.csv')

f2=open(subdir+'xgb-random-d12-e0.2-min6-tree80.csv')
f4=open(subdir+'xgb-random-d25-svc-e0.09-min6-tree100.csv')

f6=open(subdir+'xgb-random-d10-e0.2-min1-tree70.csv')
fo=open(subdir+'ave-xgb-random-online.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()

f4.readline()

f6.readline()
for c,l in enumerate(f1):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(f2.readline().split(',')[-1])
    p4=float(f4.readline().split(',')[-1])
    p6=float(f6.readline().split(',')[-1])  
    p=p1*0.35+(p2+p4+p6)/3*0.65
    
    fo=open(subdir+'ave-xgb-random-online.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()

f4.close()
fo.close()

