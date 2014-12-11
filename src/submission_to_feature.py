import pandas as pd
import numpy as np
import pickle
def submission_to_feature(sub_dir,filename,data_meta_dir,fea_name):
    sub=pd.read_csv(sub_dir+filename)
    sub=np.array(sub[['pred']])
    sub=sub.reshape((sub.shape[0]/33,33))
    sub=np.hstack([sub[:,:13],sub[:,14:]]) # remove y14
    pickle.dump(sub,open(data_meta_dir+'X_test_meta_'+fea_name+'.p','wb'))
import sys
if __name__ == "__main__":
    sub_dir=sys.argv[1]
    filename=sys.argv[2]
    data_meta_dir=sys.argv[3]
    fea_name=sys.argv[4]
    submission_to_feature(sub_dir,filename,data_meta_dir,fea_name)
