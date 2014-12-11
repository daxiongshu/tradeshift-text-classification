from tool import *
from xgb_classifier import xgb_classifier
import numpy as np
import pickle
def xgb_meta_predict(data_base_dir,data_meta_part2_dir,submission_dir):
    test_id=pickle.load(open(data_base_dir+"test_id.p","rb"))
    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
    y_part2=y_all[y_all.shape[0]/2:,:]
    
    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_numerical_part2=X_numerical[X_numerical.shape[0]/2:,:]
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    
    
    X_part2_rf=pickle.load(open(data_meta_part2_dir+ "X_meta_part2_rf.p", "rb" ) )
    X_test_rf=pickle.load(open(data_meta_part2_dir+ "X_test_meta_rf.p", "rb" ) )
    
    X_part2_svc=pickle.load(open(data_meta_part2_dir+ "X_meta_part2_svc.p", "rb" ) )
    X_test_svc=pickle.load(open(data_meta_part2_dir+ "X_test_meta_svc.p", "rb" ) )
    
    X_part2_sgd=pickle.load(open(data_meta_part2_dir+ "X_meta_part2_sgd.p", "rb" ) )
    X_test_sgd=pickle.load(open(data_meta_part2_dir+ "X_test_meta_sgd.p", "rb" ) )
    
    X_part2_best_online=pickle.load(open(data_meta_part2_dir+ "X_meta_part2_online.p", "rb" ) )
    X_test_best_online=pickle.load(open(data_meta_part2_dir+ "X_test_meta_online.p", "rb" ) )
    
    
    # private LB 0.0048854
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=18,num_round=120,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part2_best_online,X_part2_rf,X_numerical_part2]), y_part2,np.hstack([X_test_best_online, X_test_rf,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part2-d18-e0.09-min6-tree120.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0048763
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=20,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part2_best_online,X_part2_rf,X_numerical_part2]), y_part2,np.hstack([X_test_best_online, X_test_rf,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part2-d20-e0.07-min6-tree150.csv.gz', test_id , X_xgb_predict)
    
    # private LB  0.0048978
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=18,num_round=100,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part2_best_online,X_part2_rf,X_part2_svc,X_numerical_part2]), y_part2,np.hstack([X_test_best_online, X_test_rf,X_test_svc,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part2-d18-svc-e0.09-min6-tree100.csv.gz', test_id , X_xgb_predict)
    
    # private LB  0.0050270
    xgb_clf=xgb_classifier(eta=0.1,min_child_weight=6,depth=20,num_round=110,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part2_best_online,X_part2_rf,X_part2_svc,X_part2_sgd]), y_part2,np.hstack([X_test_best_online, X_test_rf,X_test_svc,X_test_sgd]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part2-d20-e0.1-min6-tree110-metaonly.csv.gz', test_id , X_xgb_predict)
   
    
    
    
    
    
    
    
    
    
    
    
    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_part2_dir=sys.argv[2]
    submission_dir=sys.argv[3]
    xgb_meta_predict(data_base_dir,data_meta_part2_dir,submission_dir)
    
    
    
