from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import pandas as pd
from deepforest import CascadeForestClassifier
import numpy as np

'''
train=pd.read_csv("train.csv",skiprows=1)
test=pd.read_csv("test.csv",skiprows=1)
print(train.shape)199999*77，第一列是id，中间75列是特征，最后一列是类别
print(test.shape)99999*76，第一列是id，后75列是特征
'''
X_train=pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=list(range(1,76)))
y_train=pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=[76])
X_test=pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/test.csv",usecols=list(range(1,76)))
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)

model = CascadeForestClassifier(random_state=1,n_jobs=4,n_trees=300,partial_mode=True,max_layers=40)
model.fit(X_train, y_train)
y_test=model.predict_proba(X_test)
y_test=pd.DataFrame(y_test)

y_test.insert(0,"id",list(range(200000,300000)))
y_test.columns=["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
print(y_test.shape)
y_test.to_csv("D:/Study/Coding/vscode-python/y_test_df.csv",index=False)
