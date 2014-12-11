from tool import *
from xgb_classifier import xgb_classifier
import numpy as np
import pickle
def xgb_meta_predict(data_base_dir,data_meta_random_dir,submission_dir):

    test_id=pickle.load(open(data_base_dir+"test_id.p","rb"))
    y_meta=pickle.load(open(data_meta_random_dir+"y_meta.p","rb"))
    
   
    X_numerical_random=pickle.load(open(data_meta_random_dir+"X_numerical_meta.p","rb"))
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    
    
    X_random_rf=pickle.load(open(data_meta_random_dir+ "X_meta_random_rf.p", "rb" ) )
    X_test_rf=pickle.load(open(data_meta_random_dir+ "X_test_meta_rf.p", "rb" ) )
    
    X_random_svc=pickle.load(open(data_meta_random_dir+ "X_meta_random_svc.p", "rb" ) )
    X_test_svc=pickle.load(open(data_meta_random_dir+ "X_test_meta_svc.p", "rb" ) )
    
  
    
    # private LB  0.0054101
    xgb_clf=xgb_classifier(eta=0.2,min_child_weight=1,depth=10,num_round=70,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_random_rf,X_random_svc,X_numerical_random]), y_meta,np.hstack([ X_test_rf,X_test_svc,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-random-d10-e0.2-min1-tree70.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0053053
    xgb_clf=xgb_classifier(eta=0.2,min_child_weight=6,depth=12,num_round=80,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_random_rf,X_random_svc,X_numerical_random]), y_meta,np.hstack([X_test_rf,X_test_svc,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-random-d12-e0.2-min6-tree80.csv.gz', test_id , X_xgb_predict)
    
    # private LB  0.0052910
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=25,num_round=100,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_random_rf,X_random_svc,X_numerical_random]), y_meta,np.hstack([X_test_rf,X_test_svc,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-random-d25-svc-e0.09-min6-tree100.csv.gz', test_id , X_xgb_predict)
    
    
    
    
    
    
    
    
    
    
    
    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_random_dir=sys.argv[2]
    submission_dir=sys.argv[3]
    xgb_meta_predict(data_base_dir,data_meta_random_dir,submission_dir)
    
    
    
