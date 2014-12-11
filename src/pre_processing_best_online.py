from best_online_model import best_online_model
from tool import *
import sys
import subprocess
def pre_processing_best_online(data_dir,data_meta_part1_dir,data_meta_part2_dir):
    split(data_dir,data_meta_part1_dir)
    best_online=best_online_model(train=data_meta_part1_dir+'train_part2.csv',label =data_meta_part1_dir+ 'trainLabels_part2.csv',test=data_meta_part1_dir+ 'train_part1.csv',
        D= 2 ** 24,alpha= .1,predict_y14=False,output_file=data_meta_part1_dir+'part1_online.csv')
    best_online.train_predict()

    
    best_online=best_online_model(train=data_meta_part1_dir+'train_part1.csv',label =data_meta_part1_dir+ 'trainLabels_part1.csv',test=data_meta_part1_dir+ 'train_part2.csv',
        D= 2 ** 24,alpha= .1,predict_y14=False,output_file=data_meta_part2_dir+'part2_online.csv')
    best_online.train_predict()
    
    
    best_online=best_online_model(data_dir+'train.csv',label = data_dir+'trainLabels.csv',test= data_dir+'test.csv',
        D= 2 ** 24,alpha= .1,predict_y14=False,output_file=data_meta_part1_dir+'best_online_test.csv')
    
    best_online.train_predict()
    cmd=' '.join(['cp',data_meta_part1_dir+'best_online_test.csv',data_meta_part2_dir ])
    subprocess.call(cmd, shell=True)
if __name__ == "__main__":
    data_dir=sys.argv[1]
    data_meta_part1_dir=sys.argv[2]
    data_meta_part2_dir=sys.argv[3]
    pre_processing_best_online(data_dir,data_meta_part1_dir,data_meta_part2_dir)
