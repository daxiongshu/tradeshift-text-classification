import pandas as pd
import numpy as np
from math import log
a=pd.read_csv("../../data-sample/testLabels.csv")
a=np.array(a[['y'+str(i) for i in range(1,34)]])
a=a.reshape((a.shape[0]*33,1))
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def totlogloss(yp,yr):
    loss=0
    c=0
    for p,y in zip(yp,yr):
        c+=1
        loss+=logloss(p,y)
    return loss/c


import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if 'csv' in f:
        x=pd.read_csv(f)
        print f,totlogloss(x['pred'],a)