from numpy.lib.shape_base import split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import optuna.integration.lightgbm as oplgb
from sklearn.model_selection import GridSearchCV

X_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=list(range(1,76))))
y_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=[76]))
X_test=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/test.csv",usecols=list(range(1,76))))


'''
train=pd.read_csv("train.csv",skiprows=1)
test=pd.read_csv("test.csv",skiprows=1)
print(train.shape)199999*77，第一列是id，中间75列是特征，最后一列是类别
print(test.shape)99999*76，第一列是id，后75列是特征
'''


#找一组最初的参数
X_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=list(range(1,76))))
y_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=[76]))
X_test=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/test.csv",usecols=list(range(1,76))))
#model=lgb.LGBMClassifier(n_estimators=120,boosting_type='dart',num_leaves=63,num_iterations=150)
def objective(trial):
    X_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=list(range(1,76))))
    y_train=np.array(pd.read_csv("D:/Study/Grade 2/Second/Machine learning/competition/train.csv",usecols=[76]))
    train_X, test_X, train_y, test_y=train_test_split(X_train, y_train, train_size=0.3)# 数据集划分
    param = {
        'n_estimators': trial.suggest_categorical('n_estimators',list(range(150,250))),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 5.0, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 5.0, 10.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.05,0.1),
        'max_depth': trial.suggest_categorical('max_depth', [5,6,7,8,9,10]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1024),  
        'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',100,2000),
    }
    model1=lgb.LGBMClassifier(**param,num_threads=4)
    model1.fit(train_X, train_y)
    pred_lgb=model1.predict(test_X)
    rmse = mean_squared_error(test_y, pred_lgb, squared=False)
    return rmse
study=optuna.create_study(direction='minimize')
n_trials=50 # try50次
study.optimize(objective, n_trials=n_trials)
params=study.best_params
print(params)

#调参
#n_estimators

params={
    'learning_rate': 0.05009346366887093, 
    'max_depth': 8, 
    'num_leaves': 20,

    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }
data_train = lgb.Dataset(X_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])#选出来177


#max_depth 和 num_leaves

model_lgb = lgb.LGBMRegressor(num_leaves=20,
                              learning_rate=0.005, n_estimators=177, max_depth=8,
                              metric='rmse', bagging_fraction = 1.0,feature_fraction = 0.42)

params_test1={
    'max_depth': range(4,10,2),
    'num_leaves':range(10,300,50)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(X_train, y_train)

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))



params_test2={
    'max_depth': [6,7,8],
    'num_leaves':[7,8,9,10,11,12]
}
model_lgb = lgb.LGBMRegressor(num_leaves=20,
                              learning_rate=0.05009346366887093, n_estimators=177, max_depth=8,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)
gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch2.fit(X_train, y_train)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))#选出来8，12



params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=12,
                              learning_rate=0.05009346366887093, n_estimators=177, max_depth=8, 
                              metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.7)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
means = gsearch3.cv_results_['mean_test_score']
params = gsearch3.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))#选出来21



params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=20)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(X_train, y_train)
means = gsearch4.cv_results_['mean_test_score']
params = gsearch4.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))#选出来1.0,0.5

#更细化feature_fraction
params_test={
    'feature_fraction': [0.42, 0.45, 0.48, 0.5, 0.52, 0.55, 0.58 ]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7, 
                              metric='rmse',  min_child_samples=20)
gsearch = GridSearchCV(estimator=model_lgb, param_grid=params_test, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch.fit(X_train, y_train)
means = gsearch.cv_results_['mean_test_score']
params = gsearch.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))#选出来0.42



params_test5={
    'reg_alpha': [9.23,9.33,9.43,9.53,9.63],
    'reg_lambda': [8.67,8.68,8.69,8.79,8.89]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=12,
                              learning_rate=0.05009346366887093, n_estimators=177, max_depth=8, 
                              metric='rmse',  min_child_samples=21, bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(X_train, y_train)
means = gsearch5.cv_results_['mean_test_score']
params = gsearch5.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))#选出来'reg_alpha': 9.63, 'reg_lambda': 8.89


#最终微调

params={
    'learning_rate':0.005,
    'n_estimators':3000,
    'max_depth':8,
    'num_leaves':24,
    'min_child_sample': 20, 
    'min_child_weigh': 0.001,
    'feature_fraction':0.24,
    'bagging_fraction':1.0,
    'reg_alpha': 9.53, 
    'reg_lambda': 8.79,
}
data_train = lgb.Dataset(X_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])



model=lgb.LGBMClassifier(   learning_rate=0.005,
    n_estimators=2244,
    max_depth=8,
    num_leaves=24,
    min_child_sample=20, 
    min_child_weigh=0.001,
    feature_fraction=0.24,
    bagging_fraction=1.0,
    reg_alpha= 9.53, 
    reg_lambda= 8.79,)
model.fit(X_train,y_train)
y_test=model.predict_proba(X_test)
y_test=pd.DataFrame(y_test)
y_test.insert(0,"id",list(range(200000,300000)))
y_test.columns=["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
print(y_test.shape)
y_test.to_csv("D:/Study/Coding/vscode-python/y_test_lgb.csv",index=False)
