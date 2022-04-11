#!/bin/bash
# author: Sajad H Alhamada 
# text_model using 'text' folder
# the below code accomedate both testing and trianing
# it generates 2 .pkl files as follow: 
# ['tfid_vect.pkl','text_model.pkl']
# these 2 filess are needed by 'myTest_text_v3.py' to run testing on new datasets

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.util import ngrams
from sklearn import metrics
import joblib,os,glob
import re

def find_file(filename, path):
    for root,dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root,filename)

def find_dir(dire,path):
    for root,dirs, files in os.walk(path):
        if dire in dirs:
            return os.path.join(root,dire)

def build_ngrams(text, n=1):
    tokens = re.findall(r"\w+", str(text.lower().split()))
    return list(ngrams(tokens, n))

# load text files
path= find_dir('text','training')
files= [file for file in glob.glob(str(path)+'/*.txt')]
df_train_x= pd.DataFrame()
for file in files:
    with open( file,'r', encoding='cp1252') as fr:
        line = fr.read()
        row={'text':str(line),'userId':str(file)[:-4].replace(path+'/','')}
        df_train_x = df_train_x.append(row,ignore_index=True)

df_train_y=pd.read_csv(find_file('profile.csv','training')).loc[:,['userid','gender']]
merged_dataset= pd.merge(left=df_train_x,right=df_train_y,left_on='userId',right_on='userid')

X = merged_dataset['text']
Y = merged_dataset['gender']

x_trian,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=7)

# represent texts using TfidfVectorizer
tfid_vect = TfidfVectorizer(analyzer=build_ngrams)

tdif_x_trian = tfid_vect.fit_transform(x_trian)
tdif_x_test = tfid_vect.transform(x_test)
joblib.dump(tfid_vect, 'tfid_vect.pkl')

# training and testing to predict gender
lr= LogisticRegression(max_iter=150)
lr.fit(tdif_x_trian,y_train)
lr_pred = lr.predict(tdif_x_test)

joblib.dump(lr, 'text_model.pkl')
lr_from_joblib = joblib.load('text_model.pkl')
lr_pred_fro_joblib= lr_from_joblib.predict(tdif_x_test)

print('Accuracy Score LR: ',metrics.accuracy_score(y_test,lr_pred))
print('Accuracy Score LR pkl: ',metrics.accuracy_score(y_test,lr_pred_fro_joblib))


