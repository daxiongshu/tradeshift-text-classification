import subprocess, sys, os, time

# the data path is changed here
# please don't forget the / at the end
# please change xgb wrapper path at ./src/xgb_classifer.py

data_dir="../data/"                      # this is the path of original data
data_base_dir="./data-base/"                    # this is the path of base features for the entire training data
data_meta_part1_dir="./data-meta-part1/"        # this is the path of meta features, if part1, 1st half of training data is used as meta data
data_meta_part2_dir="./data-meta-part2/"        # this is the path of meta features, if part2, 2nd half of training data is used as meta data
data_meta_random_dir="./data-meta-random-split/" # this is the path of meta features, training data is random split to 50/50
submission_dir="./submissions/"                 # this is the path where all solutions are generated

cmd=' '.join(['mkdir',data_base_dir,data_meta_part1_dir,data_meta_part2_dir,data_meta_random_dir,submission_dir])
subprocess.call(cmd, shell=True)


cmd=' '.join(['python', 'src/run_online.py', data_dir,submission_dir])   # run all the online models. They are self contained and don't rely on other pre-processing.
subprocess.call(cmd, shell=True)                                         

cmd=' '.join(['python', 'src/pre-ensemble.py', submission_dir])          # ensemble the predictions of all the online models
subprocess.call(cmd, shell=True)

cmd=' '.join(['python', 'src/submission_to_feature.py', submission_dir,'best_online_ensemble.csv',data_meta_part1_dir,'online_ensemble'])
subprocess.call(cmd, shell=True)                                         # use this prediction as meta feature of the test set

cmd=' '.join(['python', 'src/pre-processing-base.py', data_dir,data_base_dir]) # preprocessing base data, imputing, encoding
subprocess.call(cmd, shell=True)

cmd=' '.join(['pypy', 'src/pre_processing_best_online.py', data_dir,data_meta_part1_dir,data_meta_part2_dir]) # using online model to train base data and generate meta features
subprocess.call(cmd, shell=True)


cmd=' '.join(['python', 'src/pre-processing-meta-part1.py', data_base_dir,data_meta_part1_dir]) # using other models to generate meta features for part1, 1st half of training data
subprocess.call(cmd, shell=True)

cmd=' '.join(['python', 'src/pre-processing-meta-part2.py', data_base_dir,data_meta_part2_dir]) # using other models to generate meta features for part2, 2nd half of training data
subprocess.call(cmd, shell=True)

cmd=' '.join(['python', 'src/pre-processing-meta-random-split.py', data_base_dir,data_meta_random_dir]) # using other models to generate meta features for X_meta, which comes from a random split
subprocess.call(cmd, shell=True)


cmd=' '.join(['python', 'src/xgb_meta_part1_predict.py', data_base_dir,data_meta_part1_dir,submission_dir]) # using meta stage classifier to train, predict and generate solutions. 
subprocess.call(cmd, shell=True)                                                                            # part1 is used as the metat data. 7 models

cmd=' '.join(['python', 'src/xgb_meta_part2_predict.py', data_base_dir,data_meta_part2_dir,submission_dir]) # using meta stage classifier to train, predict and generate solutions. 
subprocess.call(cmd, shell=True)                                                                            # part2 is used as the metat data.  4 models

cmd=' '.join(['python', 'src/xgb_meta_random_split_predict.py', data_base_dir,data_meta_random_dir,submission_dir]) # using meta stage classifier to train, predict and generate solutions. 
subprocess.call(cmd, shell=True)                                                                                    # 4 models

cmd=' '.join(['python', 'src/other_model.py', data_base_dir,data_meta_part1_dir,submission_dir]) # using meta stage classifier to train, predict and generate solutions. 
subprocess.call(cmd, shell=True) 

cmd=' '.join(['gunzip ', submission_dir+'*.gz' ])
subprocess.call(cmd, shell=True)
cmd=' '.join(['python', 'src/ensemble.py', submission_dir])  # ensemble all previous prediction to generate the best_solution.csv
subprocess.call(cmd, shell=True)


