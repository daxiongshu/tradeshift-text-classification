import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgb_classifier import xgb_classifier
from best_online_model import best_online_model
from tool import *
from sklearn.svm import LinearSVC,SVC
from sklearn.cross_validation import cross_val_score, train_test_split



def pre_processing_meta_random(data_base_dir,data_meta_random_dir):
    xgb_clf=xgb_classifier(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=16,exist_prediction=True,exist_num_round=20)
    

    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
 
    

    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    X_sparse=pickle.load(open(data_base_dir+"X_sparse.p","rb"))
    X_test_sparse=pickle.load(open(data_base_dir+"X_test_sparse.p","rb"))

    X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(
        X_numerical, 
        X_sparse, 
        y_all,
        test_size = 0.5
    )



    X_meta_rf=[]
    X_meta_svc=[]



    X_test_rf=[]
    X_test_svc=[]
 
    
    for i in range(33) :
        
        predicted = None
     
        if i==13:
        
            print "%d is constant like: " % (i),"not included in meta features"
        else :
            print 'train',i
                
            y = y_base[:, i]
            rf = RandomForestClassifier(n_estimators = 150, n_jobs = 16)
            rf.fit(X_numerical_base, y)
            X_meta_rf.append(rf.predict_proba(X_numerical_meta))
            X_test_rf.append(rf.predict_proba(X_test_numerical))
    
            y = y_base[:, i]
            svm = LinearSVC()
            svm.fit(X_sparse_base, y)            
            X_meta_svc.append(svm.decision_function(X_sparse_meta))
            X_test_svc.append(svm.decision_function(X_test_sparse))

         
    
  
    X_meta_rf = np.column_stack(X_meta_rf)
    X_test_rf= np.column_stack(X_test_rf)
    pickle.dump( X_meta_rf, open(data_meta_random_dir+ "X_meta_random_rf.p", "wb" ) )
    pickle.dump( X_test_rf, open(data_meta_random_dir+ "X_test_meta_rf.p", "wb" ) )

    X_meta_svc = np.column_stack(X_meta_svc)
    X_test_svc= np.column_stack(X_test_svc)
    pickle.dump( X_meta_svc, open(data_meta_random_dir+ "X_meta_random_svc.p", "wb" ) )
    pickle.dump( X_test_svc, open(data_meta_random_dir+ "X_test_meta_svc.p", "wb" ) )

    pickle.dump( y_meta, open(data_meta_random_dir+ "y_meta.p", "wb" ) )
    pickle.dump( y_base, open(data_meta_random_dir+ "y_base.p", "wb" ) )
    pickle.dump( X_numerical_meta, open(data_meta_random_dir+ "X_numerical_meta.p", "wb" ) )
    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_random_dir=sys.argv[2]
    pre_processing_meta_random(data_base_dir,data_meta_random_dir)







