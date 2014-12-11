This code generates the 1st place solution of Tradeshift Text Classification from our team "carl and snow"

https://www.kaggle.com/c/tradeshift-text-classification

It mainly includes two kinds of models:
1) two-stage models using Xgboost and sklearn.
2) online logistic regression. 

Dependencies
Python 2.7
pypy 2.4.0
Scikit learn-0.15.2
numpy 1.7.1
scipy 0.11.0
Xgboost 0.3

To generate a solution:

1. Set Up all the dependencies
2. change the data dir in run.py
3. change the xgboost wrapper path in ./src/xgb_classifier.py
4. python run.py

The best single solution: 
xgb-part1-d18-e0.09-min6-tree120-xgb_base.csv private LB 0.0044595

The best ensemble solution: 
best-solution.csv private LB 0.0043324 (1st place)


