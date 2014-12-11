import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgb_classifier import xgb_classifier
from best_online_model import best_online_model
from tool import *



data_base_dir='./'
data_meta_part1_dir='./'
data_dir='../../data-sample/'

def pre_processing_meta_part1(data_base_dir,data_meta_part1_dir):
    
    

    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
    y_part2=y_all[y_all.shape[0]/2:,:]
 
    X_all=pickle.load(open(data_base_dir+"X_all.p","rb"))
    X_test=pickle.load(open(data_base_dir+"X_test_all.p","rb"))
    X_part1=X_all[:X_all.shape[0]/2,:]
    X_part2=X_all[X_all.shape[0]/2:,:]

    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    X_numerical_part1=X_numerical[:X_numerical.shape[0]/2,:]
    X_numerical_part2=X_numerical[X_numerical.shape[0]/2:,:]

    X_sparse=pickle.load(open(data_base_dir+"X_sparse.p","rb"))
    X_test_sparse=pickle.load(open(data_base_dir+"X_test_sparse.p","rb"))
    X_sparse_part1=X_sparse[:X_sparse.shape[0]/2,:]
    X_sparse_part2=X_sparse[X_sparse.shape[0]/2:,:]


    X_part1_xgb=[]
    X_part1_rf=[]
    X_part1_sgd=[]
  

    X_test_xgb=[]
    X_test_rf=[]
    X_test_sgd=[]
    
    # use pypy to accelerate online model
    
    X_part1_best_online=np.array(pd.read_csv(data_meta_part1_dir+'part1_online.csv')[['pred']])
    X_part1_best_online=X_part1_best_online.reshape((X_part1_best_online.shape[0]/32,32))
    X_test_best_online=np.array(pd.read_csv(data_meta_part1_dir+'best_online_test.csv')[['pred']])
    X_test_best_online=X_test_best_online.reshape((X_test_best_online.shape[0]/32,32))
    pickle.dump( X_part1_best_online, open(data_meta_part1_dir+ "X_meta_part1_online.p", "wb" ) )
    pickle.dump( X_test_best_online, open(data_meta_part1_dir+ "X_test_meta_online.p", "wb" ) )
    
    
    xgb_clf=xgb_classifier(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=16,exist_prediction=True,exist_num_round=20)
    X_part1_xgb = xgb_clf.train_predict_all_labels(X_part2, y_part2,X_part1,predict_y14=False)
    X_test_xgb = xgb_clf.train_predict_all_labels(X_all, y_all,X_test,predict_y14=False) # a little trick to make test data's meta features more accurate
    
    pickle.dump( X_part1_xgb, open(data_meta_part1_dir+ "X_meta_part1_xgb.p", "wb" ) )
    pickle.dump( X_test_xgb, open(data_meta_part1_dir+ "X_test_meta_xgb_all.p", "wb" ) )
   
    
    
    for i in range(33) :
        
        predicted = None
     
        if i==13:
        
            print "%d is constant like: " % (i),"not included in meta features"
        else :
            print 'train',i
                
            y = y_part2[:, i]
            rf = RandomForestClassifier(n_estimators=200, n_jobs=16, min_samples_leaf = 10,random_state=1,bootstrap=False,criterion='entropy',min_samples_split=5,verbose=1)
            rf.fit(X_numerical_part2, y)
            X_part1_rf.append(rf.predict_proba(X_numerical_part1))
            X_test_rf.append(rf.predict_proba(X_test_numerical))
    
            y = y_part2[:, i]
            clf=SGDClassifier(loss='log',alpha=0.000001,n_iter=100)
            clf.fit(X_sparse_part2,y)
            X_part1_sgd.append(clf.predict_proba(X_sparse_part1).T[1])
            X_test_sgd.append(clf.predict_proba(X_test_sparse).T[1])

        

         
    
  
    X_part1_rf = np.column_stack(X_part1_rf)
    X_test_rf= np.column_stack(X_test_rf)
    pickle.dump( X_part1_rf, open(data_meta_part1_dir+ "X_meta_part1_rf.p", "wb" ) )
    pickle.dump( X_test_rf, open(data_meta_part1_dir+ "X_test_meta_rf.p", "wb" ) )

    X_part1_sgd = np.column_stack(X_part1_sgd)
    X_test_sgd= np.column_stack(X_test_sgd)
    pickle.dump( X_part1_sgd, open(data_meta_part1_dir+ "X_meta_part1_sgd.p", "wb" ) )
    pickle.dump( X_test_sgd, open(data_meta_part1_dir+ "X_test_meta_sgd.p", "wb" ) )




    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_part1_dir=sys.argv[2]
    pre_processing_meta_part1(data_base_dir,data_meta_part1_dir)







