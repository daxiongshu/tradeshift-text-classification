import inspect
import os
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../xgboost-master/wrapper")

sys.path.append(code_path)
import xgboost as xgb
import numpy as np
class xgb_classifier:
    def __init__(self,eta,min_child_weight,depth,num_round,threads=8,exist_prediction=0,exist_num_round=20):
        self.eta=eta
        self.min_child_weight=min_child_weight
        self.depth=depth
        self.num_round=num_round
        self.exist_prediction=exist_prediction
        self.exist_num_round=exist_num_round  
        self.threads=threads
       
    def train_predict(self,X_train,y_train,X_test):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'binary:logistic'

        param['bst:eta'] = self.eta
        param['colsample_bytree']=1
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['eval_metric'] = 'logloss'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round
    
        bst = xgb.train( plst, xgmat_train, num_round, watchlist )
        xgmat_test = xgb.DMatrix(X_test,missing=-999)
    
        if self.exist_prediction:
        # train xgb with existing predictions
        # see more at https://github.com/tqchen/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
       
            tmp_train = bst.predict(xgmat_train, output_margin=True)
            tmp_test = bst.predict(xgmat_test, output_margin=True)
            xgmat_train.set_base_margin(tmp_train)
            xgmat_test.set_base_margin(tmp_test)
            bst = xgb.train(param, xgmat_train, self.exist_num_round, watchlist )

        ypred = bst.predict(xgmat_test)
        return ypred
        
    def train_predict_all_labels(self,X_train,y_train,X_test,predict_y14=True):
        xgb_predict=[]
        for i in range(33) :
            y = y_train[:, i]    
            predicted = None            
            if i==13:            
               if predict_y14:
                  predicted = np.ones(X_test.shape[0]) * 0 
               else:
                  continue
               print "%d is constant like: " % (i)
            else :
               print i, 'train xgb'    
               predicted =self.train_predict(X_train, y,X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)
    def train_predict_label(self,X_train,y_train,X_test,label):
        xgb_predict=[]
        for i in range(label-1,label) :
            y = y_train[:, i]    
            predicted = None            
            if i==13:            
               if predict_y14:
                  predicted = np.ones(X_test.shape[0]) * 0 
               else:
                  continue
               print "%d is constant like: " % (i)
            else :
               print i, 'train xgb'    
               predicted =self.train_predict(X_train, y,X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)


