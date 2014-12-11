import sys
subdir=sys.argv[1]
f1=open(subdir+'xgb-part2-d18-e0.09-min6-tree120.csv')
f2=open(subdir+'ave-xgb-random-online.csv')
f3=open(subdir+'xgb-part2-d20-e0.07-min6-tree150.csv')
f4=open(subdir+'xgb-part2-d18-svc-e0.09-min6-tree100.csv')
f5=open(subdir+'xgb-part2-d20-e0.1-min6-tree110-metaonly.csv')
fo=open(subdir+'ave-xgb-part2-random-online.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()
f3.readline()
f4.readline()
f5.readline()
c=0
for l,m,n in zip(f1,f2,f3):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    p3=float(n.split(',')[-1])  
    p4=float(f4.readline().split(',')[-1])
    p5=float(f5.readline().split(',')[-1])
#    p=(p2*0.55)*0.4+(p1+p3+p4)*0.6/3
    p=p2*0.4+(p1+p3+p4+p5*0.5)/3.5*0.6
    fo=open(subdir+'ave-xgb-part2-random-online.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()
f3.close()
fo.close()
f4.close()
f5.close()
