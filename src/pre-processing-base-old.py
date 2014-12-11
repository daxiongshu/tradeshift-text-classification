import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle
from scipy import sparse
#from counter_old import *
data_dir='../../data-sample/'
data_base_dir='./'
import sys
def pre_processing_base(data_dir, data_base_dir):

    
    train = pd.read_csv(data_dir + 'train.csv')
    test = pd.read_csv(data_dir + 'test.csv')
    pickle.dump(np.array(test[['id']]),open(data_base_dir+"test_id.p","wb"))

    # encode 10 hashed features into sparse matrix 
    vec = DictVectorizer()
    names_categorical = []

    for name in train.columns :    
        if name.startswith('x') :
            column_type, _ = max(Counter(map(lambda x: str(type(x)), train[name])).items(), key = lambda x: x[1])
        
        # LOL expression
            if column_type == str(str) :
                train[name] = map(str, train[name])
                test[name] = map(str, test[name])

                names_categorical.append(name)
                print name, len(np.unique(train[name]))
     

    X_sparse = vec.fit_transform(train[names_categorical].T.to_dict().values())
    X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

    # pre-process the rest features as numerical features
    numerical_label=['x'+str(i) for i in range(1,146) if 'x'+str(i) not in names_categorical]

    X_numerical=train[numerical_label]
    X_numerical.replace('YES', 1, inplace = True)
    X_numerical.replace('NO', 0, inplace = True)
    X_numerical=np.array(X_numerical).astype(float)
    for c,i in enumerate(X_numerical.T):
        i[np.isnan(i)]=-999
        X_numerical[:,c]=i


    X_test_numerical=test[numerical_label]
    X_test_numerical.replace('YES', 1, inplace = True)
    X_test_numerical.replace('NO', 0, inplace = True)
    X_test_numerical=np.array(X_test_numerical)

    for c,i in enumerate(X_test_numerical.T):
        i[np.isnan(i)]=-999
        X_test_numerical[:,c]=i

    print "pre-process done.",X_numerical.shape, X_sparse.shape, X_test_numerical.shape, X_test_sparse.shape

    # read labels
    y_labels=np.array(pd.read_csv(data_dir + 'trainLabels.csv'))[:,1:] # remove id
    pickle.dump( y_labels, open( data_base_dir+ "y.p", "wb" ) )

    pickle.dump( X_numerical, open(data_base_dir+  "X_numerical.p", "wb" ) )
    pickle.dump( X_sparse, open(data_base_dir+  "X_sparse.p", "wb" ) )

    pickle.dump( X_test_numerical, open(data_base_dir+  "X_test_numerical.p", "wb" ) )
    pickle.dump(X_test_sparse , open( data_base_dir+ "X_test_sparse.p", "wb" ) )

    # store the split version of data, for two-stage classifiers
   
    X_all=sparse.csr_matrix(sparse.hstack([sparse.coo_matrix(X_sparse),sparse.coo_matrix(X_numerical)]))
    pickle.dump(X_all,open(data_base_dir+ "X_all.p","wb"))

    
    X_test_all=sparse.csr_matrix(sparse.hstack([sparse.coo_matrix(X_test_sparse),sparse.coo_matrix(X_test_numerical)]))
    pickle.dump(X_test_all,open(data_base_dir+ "X_test_all.p","wb"))

if __name__ == "__main__":
    data_dir=sys.argv[1]
    data_base_dir=sys.argv[2]
    pre_processing_base(data_dir, data_base_dir)




