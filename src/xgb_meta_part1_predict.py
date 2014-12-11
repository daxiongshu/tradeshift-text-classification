from tool import *
from xgb_classifier import xgb_classifier
import numpy as np
import pickle
def xgb_meta_predict(data_base_dir,data_meta_part1_dir,submission_dir):
    test_id=pickle.load(open(data_base_dir+"test_id.p","rb"))
    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
    y_part1=y_all[:y_all.shape[0]/2,:]
    
    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_numerical_part1=X_numerical[:X_numerical.shape[0]/2,:]
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    
    X_part1_xgb=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_xgb.p", "rb" ) )
    X_test_xgb =pickle.load(open(data_meta_part1_dir+ "X_test_meta_xgb_all.p", "rb" ) )
    
    X_part1_rf=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_rf.p", "rb" ) )
    X_test_rf=pickle.load(open(data_meta_part1_dir+ "X_test_meta_rf.p", "rb" ) )
    
    X_part1_sgd=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_sgd.p", "rb" ) )
    X_test_sgd=pickle.load(open(data_meta_part1_dir+ "X_test_meta_sgd.p", "rb" ) )
    
    X_part1_best_online=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_online.p", "rb" ) )
    X_test_best_online=pickle.load(open(data_meta_part1_dir+ "X_test_meta_online.p", "rb" ) )
    X_test_online_ensemble=pickle.load(open(data_meta_part1_dir+ "X_test_meta_online_ensemble.p", "rb" ) )
    
    
    # best single model submitted, private LB 0.0044595, X_test_meta 
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=18,num_round=120,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_part1_xgb,X_numerical_part1]), y_part1,np.hstack([X_test_online_ensemble, X_test_rf,X_test_sgd,X_test_xgb,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d18-e0.09-min6-tree120-xgb_base.csv.gz', test_id , X_xgb_predict)
    
    # best single model (not submitted by itself), private LB 0.0044591, not submitted alone
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=20,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_part1_xgb,X_numerical_part1]), y_part1,np.hstack([X_test_online_ensemble, X_test_rf,X_test_sgd,X_test_xgb,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d20-e0.07-min6-tree150-xgb_base.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0047360 correct! try "boosting from existing predictions"
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=20,num_round=20,threads=16,exist_prediction=True,exist_num_round=150) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_numerical_part1]), y_part1,np.hstack([X_test_best_online, X_test_rf,X_test_sgd,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d20-e0.07-min6-tree20-extree-150.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0047103, 
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=18,num_round=1,threads=16,exist_prediction=True,exist_num_round=120) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_numerical_part1]), y_part1,np.hstack([X_test_online_ensemble, X_test_rf,X_test_sgd,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d18-e0.09-min6-tree1-extree-120.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0047000, using ensembled online predictions as meta feature for test sets!
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=20,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_numerical_part1]), y_part1,np.hstack([X_test_online_ensemble, X_test_rf,X_test_sgd,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d20-e0.07-min6-tree150.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0047313, correct!
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=19,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_numerical_part1]), y_part1,np.hstack([X_test_best_online, X_test_rf,X_test_sgd,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d19-e0.07-min6-tree150.csv.gz', test_id , X_xgb_predict)
    
    # private LB 0.0047446, correct!
    xgb_clf=xgb_classifier(eta=0.09,min_child_weight=6,depth=18,num_round=120,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(np.hstack([X_part1_best_online,X_part1_rf,X_part1_sgd,X_numerical_part1]), y_part1,np.hstack([X_test_best_online, X_test_rf,X_test_sgd,X_test_numerical]),predict_y14=True)
    save_predictions(submission_dir+'xgb-part1-d18-e0.09-min6-tree120.csv.gz', test_id , X_xgb_predict)
    
    
    
    
    
    
    
    
    
    
    
    
    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_part1_dir=sys.argv[2]
    submission_dir=sys.argv[3]
    xgb_meta_predict(data_base_dir,data_meta_part1_dir,submission_dir)
    
    
    
