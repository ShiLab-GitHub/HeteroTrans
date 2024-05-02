import pandas as pd
from sklearn.model_selection import KFold

data = pd.read_csv('./SAMPL.csv')

kf = KFold(n_splits=10)

num = 0

for train,test in kf.split(data):
    train_set = data.iloc[train]
    test_set = data.iloc[test]
    train_set.to_csv("2022_fold_{}_train.csv".format(num),index=None)
    test_set.to_csv("2022_fold_{}_test.csv".format(num),index=None)
    test_set.to_csv("2022_fold_{}_valid.csv".format(num),index=None)
    num = num+1