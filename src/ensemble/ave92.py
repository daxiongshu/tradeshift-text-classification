import sys
subdir=sys.argv[1]
f1=open(subdir+'xgb-part1-d18-e0.09-min6-tree1-extree-120.csv')
f2=open(subdir+'ave-xgb-part2-random-online.csv')
f3=open(subdir+'xgb-part1-d20-e0.07-min6-tree150.csv')
f4=open(subdir+'xgb-part1-d18-e0.09-min6-tree120.csv')
f5=open(subdir+'xgb-part1-d19-e0.07-min6-tree150.csv')
f6=open(subdir+'xgb-part1-d20-e0.07-min6-tree20-extree-150.csv')

fo=open(subdir+'ave-xgb-part1-part2-random-online.csv','a')
fo.write(f1.readline())
fo.close()
f2.readline()
f3.readline()
f4.readline()
f5.readline()
f6.readline()

c=0
for l,m in zip(f1,f2):
    idx=l.split(',')[0]
    p1=float(l.split(',')[-1])
    p2=float(m.split(',')[-1])
    p3=float(f3.readline().split(',')[-1])
    p4=float(f4.readline().split(',')[-1])
    p5=float(f5.readline().split(',')[-1])
    p6=float(f6.readline().split(',')[-1])

    p=p2*0.6+(p1+p3+p4+p5+p6)*0.4/5
    
    fo=open(subdir+'ave-xgb-part1-part2-random-online.csv','a')
    fo.write(idx+','+str(p)+'\n')
    fo.close()
    c+=1
    if c%100000==0:
       print c,'processed'
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

fo.close()
