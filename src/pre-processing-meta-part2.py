import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgb_classifier import xgb_classifier
from best_online_model import best_online_model
from tool import *
from sklearn.svm import LinearSVC,SVC




def pre_processing_meta_part2(data_base_dir,data_meta_part2_dir):
    xgb_clf=xgb_classifier(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=16,exist_prediction=True,exist_num_round=20)
    

    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
    y_part1=y_all[:y_all.shape[0]/2,:]
 
    

    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    X_numerical_part1=X_numerical[:X_numerical.shape[0]/2,:]
    X_numerical_part2=X_numerical[X_numerical.shape[0]/2:,:]

    X_sparse=pickle.load(open(data_base_dir+"X_sparse.p","rb"))
    X_test_sparse=pickle.load(open(data_base_dir+"X_test_sparse.p","rb"))
    X_sparse_part1=X_sparse[:X_sparse.shape[0]/2,:]
    X_sparse_part2=X_sparse[X_sparse.shape[0]/2:,:]



    X_part2_rf=[]
    X_part2_svc=[]
    X_part2_sgd=[]


    X_test_rf=[]
    X_test_svc=[]
    X_test_sgd=[]
    
    # use pypy to accelerate online model
    
    X_part2_best_online=np.array(pd.read_csv(data_meta_part2_dir+'part2_online.csv')[['pred']])
    X_part2_best_online=X_part2_best_online.reshape((X_part2_best_online.shape[0]/32,32))

    pickle.dump( X_part2_best_online, open(data_meta_part2_dir+ "X_meta_part2_online.p", "wb" ) )
    
    X_test_best_online=np.array(pd.read_csv(data_meta_part2_dir+'best_online_test.csv')[['pred']])
    X_test_best_online=X_test_best_online.reshape((X_test_best_online.shape[0]/32,32))
   
    pickle.dump( X_test_best_online, open(data_meta_part2_dir+ "X_test_meta_online.p", "wb" ) )
    
   
    
    
    for i in range(33) :
        
        predicted = None
     
        if i==13:
        
            print "%d is constant like: " % (i),"not included in meta features"
        else :
            print 'train',i
                
            y = y_part1[:, i]
            rf = RandomForestClassifier(n_estimators=200, n_jobs=16, min_samples_leaf = 10,random_state=1,bootstrap=False,criterion='entropy',min_samples_split=5,verbose=1)
            rf.fit(X_numerical_part1, y)
            X_part2_rf.append(rf.predict_proba(X_numerical_part2))
            X_test_rf.append(rf.predict_proba(X_test_numerical))
    
            y = y_part1[:, i]
            svm = LinearSVC(C=0.17)
            svm.fit(X_sparse_part1, y)            
            X_part2_svc.append(svm.decision_function(X_sparse_part2))
            X_test_svc.append(svm.decision_function(X_test_sparse))

            y = y_part1[:, i]
            clf=SGDClassifier(loss='log',alpha=0.000001,n_iter=100)
            clf.fit(X_sparse_part1,y)
            X_part2_sgd.append(clf.predict_proba(X_sparse_part2).T[1])
            X_test_sgd.append(clf.predict_proba(X_test_sparse).T[1])

         
    
  
    X_part2_rf = np.column_stack(X_part2_rf)
    X_test_rf= np.column_stack(X_test_rf)
    pickle.dump( X_part2_rf, open(data_meta_part2_dir+ "X_meta_part2_rf.p", "wb" ) )
    pickle.dump( X_test_rf, open(data_meta_part2_dir+ "X_test_meta_rf.p", "wb" ) )

    X_part2_svc = np.column_stack(X_part2_svc)
    X_test_svc= np.column_stack(X_test_svc)
    pickle.dump( X_part2_svc, open(data_meta_part2_dir+ "X_meta_part2_svc.p", "wb" ) )
    pickle.dump( X_test_svc, open(data_meta_part2_dir+ "X_test_meta_svc.p", "wb" ) )

    X_part2_sgd = np.column_stack(X_part2_sgd)
    X_test_sgd= np.column_stack(X_test_sgd)
    pickle.dump( X_part2_sgd, open(data_meta_part2_dir+ "X_meta_part2_sgd.p", "wb" ) )
    pickle.dump( X_test_sgd, open(data_meta_part2_dir+ "X_test_meta_sgd.p", "wb" ) )


    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_part2_dir=sys.argv[2]
    pre_processing_meta_part2(data_base_dir,data_meta_part2_dir)







