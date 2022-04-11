#!/bin/bash
# author: Sajad H Alhamada 
# text_model using LIWC.csv file
# the below code accomedate both testing and trianing
# it generates 10 .pkl files as follow:
# ['support_e.pkl','text_model_e.pkl',
# 'support_n.pkl','text_model_n.pkl',
# 'support_a.pkl','text_model_a.pkl',
# 'support_o.pkl','text_model_o.pkl',
# 'support_c.pkl','text_model_c.pkl']
# these 10 files are needed by 'myTest_text_v3.py' to run testing on new datasets
# You might get UserWarning running this file 'text_model_v2.py', if so please ignore it. It is sklearn bug that does NOT effect the results.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn import metrics
import joblib,os,glob


def find_file(filename, path):
    for root,dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root,filename)

def find_dir(dire,path):
    for root,dirs, files in os.walk(path):
        if dire in dirs:
            return os.path.join(root,dire)

# load liwc.csv file and profile.csv file
df_train_x= pd.read_csv(find_file('LIWC.csv','training'))
df_train_y=pd.read_csv(find_file('profile.csv','training'))
# merging them into one dataframe
merged_dataset= pd.merge(left=df_train_x,right=df_train_y,left_on='userId',right_on='userid')
drop_col= list (df_train_y.columns)

Y = merged_dataset[drop_col]
drop_col.append('userId')
drop_col.append('Seg')
X = merged_dataset.drop(drop_col,axis=1)

x_trian,x_test,y_train,y_test= train_test_split(X,Y,test_size=1500,random_state=7)

# algorithm used
lr= LinearRegression()

# to select features using Recursive Feature Elimination
rfecv_lr = RFECV(
    estimator=lr,
    min_features_to_select=5,scoring='r2'
)

# traing and test for open trait
rfecv_lr.fit(x_trian, y_train['ope'])
joblib.dump(rfecv_lr.support_, 'support_o.pkl')
from_joblib = joblib.load('support_o.pkl')
lr.fit(x_trian.loc[:,from_joblib],y_train['ope'])
joblib.dump(lr, 'text_model_o.pkl')
lr_from_joblib = joblib.load('text_model_o.pkl')
lr_pred = lr_from_joblib.predict(x_test.loc[:,from_joblib])
print('MSE Score LR o: ',metrics.mean_squared_error(y_test['ope'],lr_pred))

# traing and test for con trait
rfecv_lr.fit(x_trian, y_train['con'])
joblib.dump(rfecv_lr.support_, 'support_c.pkl')
from_joblib = joblib.load('support_c.pkl')
lr.fit(x_trian.loc[:,from_joblib],y_train['con'])
joblib.dump(lr, 'text_model_c.pkl')
lr_from_joblib = joblib.load('text_model_c.pkl')
lr_pred = lr_from_joblib.predict(x_test.loc[:,from_joblib])
print('MSE Score LR c: ',metrics.mean_squared_error(y_test['con'],lr_pred))

# traing and test for ext trait
rfecv_lr.fit(x_trian, y_train['ext'])
joblib.dump(rfecv_lr.support_, 'support_e.pkl')
from_joblib = joblib.load('support_e.pkl')
lr.fit(x_trian.loc[:,from_joblib],y_train['ext'])
joblib.dump(lr, 'text_model_e.pkl')
lr_from_joblib = joblib.load('text_model_e.pkl')
lr_pred = lr_from_joblib.predict(x_test.loc[:,from_joblib])
print('MSE Score LR e: ',metrics.mean_squared_error(y_test['ext'],lr_pred))

# traing and test for agr trait
rfecv_lr.fit(x_trian, y_train['agr'])
joblib.dump(rfecv_lr.support_, 'support_a.pkl')
from_joblib = joblib.load('support_a.pkl')
lr.fit(x_trian.loc[:,from_joblib],y_train['agr'])
joblib.dump(lr, 'text_model_a.pkl')
lr_from_joblib = joblib.load('text_model_a.pkl')
lr_pred = lr_from_joblib.predict(x_test.loc[:,from_joblib])
print('MSE Score LR a: ',metrics.mean_squared_error(y_test['agr'],lr_pred))

# traing and test for neu trait
rfecv_lr.fit(x_trian, y_train['neu'])
joblib.dump(rfecv_lr.support_, 'support_n.pkl')
from_joblib = joblib.load('support_n.pkl')
lr.fit(x_trian.loc[:,from_joblib],y_train['neu'])
joblib.dump(lr, 'text_model_n.pkl')
lr_from_joblib = joblib.load('text_model_n.pkl')
lr_pred = lr_from_joblib.predict(x_test.loc[:,from_joblib])
print('MSE Score LR n: ',metrics.mean_squared_error(y_test['neu'],lr_pred))



