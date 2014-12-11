from tool import *
from xgb_classifier import xgb_classifier
import numpy as np
import pickle
from scipy import sparse
def xgb_meta_predict(data_base_dir,data_meta_part1_dir,submission_dir):
    test_id=pickle.load(open(data_base_dir+"test_id.p","rb"))
    y_all=pickle.load(open(data_base_dir+"y.p","rb"))
    X_all=pickle.load(open(data_base_dir+"X_all.p","rb"))
    X_test=pickle.load(open(data_base_dir+"X_test_all.p","rb"))
    y_part1=y_all[:y_all.shape[0]/2,:]
    
    
    xgb_clf=xgb_classifier(eta=0.07,min_child_weight=6,depth=20,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_all_labels(X_all, y_all,X_test,predict_y14=True)
    save_predictions(submission_dir+'xgb-raw-d20-e0.07-min6-tree150.csv.gz', test_id , X_xgb_predict)
    
    xgb_clf=xgb_classifier(eta=0.1,min_child_weight=7,depth=100,num_round=150,threads=16) 
    X_xgb_predict = xgb_clf.train_predict_label(X_all, y_all,X_test,label=33) # predict label 33 only
    save_predictions(submission_dir+'xgb-y33-d100-e0.1-min7-tree150.csv.gz', test_id , X_xgb_predict)
    
    
    X_part1_best_online=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_online.p", "rb" ) )
    X_test_best_online=pickle.load(open(data_meta_part1_dir+ "X_test_meta_online.p", "rb" ) )
    
    X_numerical=pickle.load(open(data_base_dir+"X_numerical.p","rb"))
    X_numerical_part1=X_numerical[:X_numerical.shape[0]/2,:]
    X_test_numerical=pickle.load(open(data_base_dir+"X_test_numerical.p","rb"))
    
    X_part1_xgb=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_xgb.p", "rb" ) )
    X_test_xgb =pickle.load(open(data_meta_part1_dir+ "X_test_meta_xgb_all.p", "rb" ) )
    
    X_part1_rf=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_rf.p", "rb" ) )
    X_test_rf=pickle.load(open(data_meta_part1_dir+ "X_test_meta_rf.p", "rb" ) )
    
    X_part1_sgd=pickle.load(open(data_meta_part1_dir+ "X_meta_part1_sgd.p", "rb" ) )
    X_test_sgd=pickle.load(open(data_meta_part1_dir+ "X_test_meta_sgd.p", "rb" ) )
    
    X_sparse=pickle.load(open(data_base_dir+"X_sparse.p","rb"))
    X_test_sparse=pickle.load(open(data_base_dir+"X_test_sparse.p","rb"))
    X_sparse_part1=X_sparse[:X_sparse.shape[0]/2,:]
    
    X=sparse.csr_matrix(sparse.hstack((X_sparse_part1,sparse.coo_matrix(np.hstack  ([X_part1_best_online,X_part1_rf,X_part1_sgd,X_part1_xgb,X_numerical_part1]).astype(float)))))
    Xt=sparse.csr_matrix(sparse.hstack((X_test_sparse,sparse.coo_matrix(np.hstack  ([X_test_best_online,X_test_rf,X_test_sgd,X_test_xgb,X_test_numerical]).astype(float)))))
    xgb_clf=xgb_classifier(eta=0.1,min_child_weight=6,depth=30,num_round=80,threads=16)
    X_xgb_predict = xgb_clf.train_predict_label(X, y_part1,Xt,label=33) # predict label 33 only
    save_predictions(submission_dir+'xgb-y33-d30-e0.1-min6-tree80-all-sparse.csv.gz', test_id , X_xgb_predict)
    
import sys
if __name__ == "__main__":
    data_base_dir=sys.argv[1]
    data_meta_part1_dir=sys.argv[2]
    submission_dir=sys.argv[3]
    xgb_meta_predict(data_base_dir,data_meta_part1_dir,submission_dir)
    
