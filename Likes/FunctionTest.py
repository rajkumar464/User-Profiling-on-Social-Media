'''
TCSS 555 Winter 2022  Project
Functions to input data
Author: Zhengwu Liu
'''

import getopt
import sys, csv
import pandas as pd
import numpy as py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn import datasets
import xml.dom.minidom as Dom
import pickle
import errno
import os

#inputFile = "B:\\onedrive\\UWT\\WIN 2022\\555\\tcss555\\training"
#inputTest = "B:\\onedrive\\UWT\\WIN 2022\\555\\tcss555\\public-test-data\\"
#outputFile = "B:\\onedrive\\UWT\\WIN 2022\\555\\tcss555\\output\\"

#outputFile = "/home/alanliu/output/"

opts,args = getopt.getopt(sys.argv[1:], "i:o:")
inputTest = ""
outputFile = ""
trainingFile = 'data/training'
for op,value in opts:
    if op == "-i":
        inputTest = value
        print(inputTest)
    if op == "-o":
        outputFile = value
        print(outputFile)

inputFile = "/data/training"
inputTest = "/data/public-test-data"

def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def groupAge(age):
    if age <= 24:
        return 0
    if age > 24 and age <= 34:
        return 1
    if age > 34 and age <= 49:
        return 2
    if age > 49:
        return 3

def outputResult(singleUser, outputFile):

    doc = Dom.Document()
    root_node = doc.createElement("")
    root_node.setAttribute("user id", singleUser.id)
    root_node.setAttribute("age_group", singleUser.age_group)
    root_node.setAttribute("gender", str(singleUser.gender))
    root_node.setAttribute("extrovert", str(singleUser.extrovert))
    root_node.setAttribute("neurotic", str(singleUser.neurotic))
    root_node.setAttribute("agreeable", str(singleUser.agreeable))
    root_node.setAttribute("conscientious", str(singleUser.conscientious))
    root_node.setAttribute("open", str(singleUser.open))
    doc.appendChild(root_node)

    f = open(outputFile +singleUser.id+ ".xml" , "w")
    root_node.writexml(f,addindent='', newl='\n')
    f.close()



    return

class userDataStruct():
    userData = ""
    likesData = ""
    featureData = ""
    textData = 0

def likesDataprepocessing(userDF, userTestDF, likesize):
    # userDF.featureData.rename(columns={'userId':'userid'},inplace=True)

    #print(likesize)
    likesData = pd.merge(userDF.userData, userDF.likesData, on='userid', how='right')
    likesData = likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > likesize)
    likesData = likesData.reindex()
    # print(likesData.index)
    userlikegroup = likesData.groupby('like_id').groups
    useridgroup = likesData.groupby('userid').groups
    likesData_predict = pd.merge(userTestDF.userData, userTestDF.likesData, on='userid', how='right')
    useridTestGroup = likesData_predict.groupby('userid').groups
    userlikegroup_test = likesData_predict.groupby('like_id').groups

    newfeature = pd.DataFrame(columns=userlikegroup.keys())
    feature_predict1 = pd.DataFrame(columns=userlikegroup.keys())
    feature_predict1['userid'] = userTestDF.userData['userid']
    feature_predict2 = pd.DataFrame(columns=userlikegroup_test.keys())
    feature_predict2['userid'] = userTestDF.userData['userid']
    newfeature['userid'] = userDF.userData['userid']
    newfeature['gender'] = userDF.userData['gender']
    newfeature['age_group'] = userDF.userData['age'].apply(groupAge)

    newfeature = newfeature.fillna(0)
    feature_predict1 = feature_predict1.fillna(0)
    #print("orig test like:")
    #print(feature_predict1)
    feature_predict2 = feature_predict2.fillna(0)
    feature_predict = pd.concat([feature_predict1, feature_predict2[feature_predict2.columns.difference(feature_predict1.columns)]], axis=1)
    # feature_predict = feature_predict.loc[:, feature_predict.columns[0:-1]]

    #print("test likes:")
    #print(feature_predict2)
    #print(feature_predict)

    # print(newfeature)

    # print(likesData.loc[0,['userid']])
    #print("!!!")
    # print(useridgroup)
    # print(useridTestGroup)
    #print(likesData)
    # print(likesData.columns)
    # print(likesData.columns.size)
    #print(likesData_predict)
    #print("!!!")
    #print("111")
    #print(useridgroup)
    #print(useridTestGroup)
    #print(newfeature)
    #print(feature_predict)

    for key in useridgroup.keys():
        # print(key)
        iuserid = key
        tempindex = useridgroup[iuserid]
        target = likesData.loc[tempindex, ['like_id']]  # .reindex()
        newfeature.loc[newfeature['userid'] == iuserid, target['like_id']] = 1


    for key in useridTestGroup.keys():
        #print(key)
        iuserid = key
        tempindex = useridTestGroup[iuserid]
        target = likesData_predict.loc[tempindex, ['like_id']]
        #print(target)
        feature_predict.loc[feature_predict['userid'] == iuserid, target['like_id']] = 1

    return newfeature , feature_predict.loc[:, feature_predict.columns[0:newfeature.columns.size-3]]




userDF = userDataStruct()
userDF.userData = pd.read_csv(inputFile + "/profile/profile.csv")
userDF.likesData = pd.read_csv(inputFile + "/relation/relation.csv")
userDF.featureData = pd.read_csv(inputFile + "/LIWC/LIWC.csv")

userTestDF = userDataStruct()
userTestDF.userData = pd.read_csv(inputTest + "profile/profile.csv")
userTestDF.likesData = pd.read_csv(inputTest + "relation/relation.csv")
userTestDF.featureData = pd.read_csv(inputTest + "LIWC/LIWC.csv")

#print("user df: ")
#print(userDF.userData)
#print(userTestDF.userData)




likesize = 50
likesdata, likesdata_predict = likesDataprepocessing(userDF, userTestDF, likesize)
#likesdata = likesDataprepocessing(userDF, userTestDF, likesize)

#print("likesdata: ")
#print(likesdata)
#print(likesdata.shape)
#print(likesdata_predict)
# print(likesdata_test.columns[0:-3])
# print(likesdata_test.shape)

X = likesdata.loc[:, likesdata.columns[0:-3]]
print("X::")
print(X.shape)
print(X)
#print(likesdata.columns)
yg = likesdata['gender']
ya = likesdata['age_group']
#print(yg)

MNB = MultinomialNB()
LR = LogisticRegression()  # 0.74,0.544
CART = tree.DecisionTreeClassifier()  # 0.658 0.49
BOOST = ensemble.AdaBoostClassifier()  # 0.699 0.52
SVC = LinearSVC()
# userTestDF
# userTestDF.likesData.groupby('userid')
X_test = likesdata_predict
#print("X_test::::")
#print(X_test)

LR.fit(X, yg)
filename = 'B:\\finalized_model.sav'
pickle.dump(LR, open(filename, 'wb'))

y_gender_predict = LR.predict(X_test)
#print("predicted gender:")
#print(y_gender_predict)

LR.fit(X, ya)
y_age_group_predict = LR.predict(X_test)
#print("predicted age group:")
#print(y_age_group_predict)

result_df = userTestDF.userData.drop(columns='age').drop(columns='gender')
result_df.insert(2, 'age_group', y_age_group_predict)
result_df.insert(3, 'gender', y_gender_predict)


print("result: ")
print(result_df)

make_dir(outputFile)
i = 0;
for row in result_df.iterrows():
    print(i)
    i = i + 1
    doc = Dom.Document()
    root_node = doc.createElement("")
    gender = row[1]['gender']
    age_group = row[1]['age_group']
    if gender == 1.0:
        gender = "female"
    else:
        gender = "male"

    if age_group == 0:
        age_group = "xx-24"
    if age_group == 1:
        age_group = "25-34"
    if age_group == 2:
        age_group = "35-49"
    if age_group == 3:
        age_group = "49-xx"

    root_node.setAttribute("user id", row[1]['userid'])
    root_node.setAttribute("age_group", age_group)
    root_node.setAttribute("gender", gender)
    root_node.setAttribute("extrovert", "")
    root_node.setAttribute("neurotic", "")
    root_node.setAttribute("agreeable", "")
    root_node.setAttribute("conscientious", "")
    root_node.setAttribute("open", "")
    doc.appendChild(root_node)

    f = open(outputFile + row[1]['userid'] + ".xml" , "w")
    root_node.writexml(f,addindent='', newl='\n')
    f.close()


print('MNB')
print(cross_val_score(MNB, X, yg, cv=10).sum() / 10)
print(cross_val_score(MNB, X, ya, cv=10).sum() / 10)
print('LR')
print(cross_val_score(LR, X, yg, cv=10).sum() / 10)
print(cross_val_score(LR, X, ya, cv=10).sum() / 10)
print('CART')
print(cross_val_score(CART,X,yg,cv=10).sum()/10)
print(cross_val_score(CART,X,ya,cv=10).sum()/10)
print('BOOST')
print(cross_val_score(BOOST,X,yg,cv=10).sum()/10)
print(cross_val_score(BOOST,X,ya,cv=10).sum()/10)
print('SVC')
print(cross_val_score(SVC,X,yg,cv=10).sum()/10)
print(cross_val_score(SVC,X,ya,cv=10).sum()/10)