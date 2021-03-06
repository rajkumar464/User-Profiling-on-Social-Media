'''
TCSS 555 Winter 2022  Project
Functions to input data
Author: Zhengwu Liu
'''

import sys,csv
import pandas as pd
import numpy as np

from userData import user
from sklearn import datasets
from learningTools import groupAge

'''
Struct to store different user data
userData : user profile data csv
likesData : likeid data csv
featureData : user LIWC data csv
textData : user text data
'''
class userDataStruct():
    userData = ""
    likesData = ""
    featureData = ""
    textData = 0

'''
Read user's profile data
Input:
    inputFile: root folder of training/testing data
Output:
    a list, list of users, every user's data is stored in user struct
'''
def sampleInput(inputFile):
    
    inputFilePro = inputFile + "/profile/profile.csv"
    
    csvReader = csv.reader(open(inputFilePro))
    
    count = 0
    users = []
    
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
        
        singleUser = user(1)
        singleUser.id = parameters[1]
        singleUser.age = float(parameters[2])
        singleUser.genderType = float(parameters[3])
        singleUser.open = float(parameters[4])
        singleUser.conscientious = float(parameters[5])
        singleUser.extrovert = float(parameters[6])
        singleUser.agreeable = float(parameters[7])
        singleUser.neurotic = float(parameters[8])
                
        user.idList.append(parameters[1])
        users.append(singleUser)
        
        print(len(users))
    
    inputFileRel = inputFile + "/relation/relation.csv"
    
    csvReader = csv.reader(open(inputFileRel))
    count = 0
    userID = -1
    '''
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue       
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
                               
        userID = parameters[1]
        print(parameters[1])
        users[user.idList.index(userID)].likeID.append(parameters[2])
    
    '''
    return users

'''
Read user's like data
Input:
    inputFile: root folder of training/testing data
Output:
    a list, list of users, only user's id and liked id is stored in user struct
'''
def userInput(inputFile):
    
    inputFilePro = inputFile + "/profile/profile.csv"
    
    csvReader = csv.reader(open(inputFilePro))
    
    count = 0
    users = []
    
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
        
        singleUser = user(1)
        singleUser.id = parameters[1]
        user.idList.append(parameters[1])
        users.append(singleUser)
    
    
    inputFileRel = inputFile + "/relation/relation.csv"
    
    csvReader = csv.reader(open(inputFileRel))
    count = 0
    userID = -1
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue       
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
                               
        userID = parameters[1]
        users[user.idList.index(userID)].likeID.append(parameters[2])
    
    return users

'''
Read raw data of each csv
Input:
    inputFile: root folder of input (training/testing data)
Output:
    userDataStruct, struct contains all user's Data
'''
def sampleInputPd(inputFile):
    userDF = userDataStruct()
    userDF.userData = pd.read_csv(inputFile+"/profile/profile.csv")
    userDF.likesData = pd.read_csv(inputFile+"/relation/relation.csv")
    userDF.featureData = pd.read_csv(inputFile+"/LIWC/LIWC.csv")
    userDF.textData = datasets.load_files(inputFile,load_content=True,categories='text')
    
    return userDF

'''
Generate a dataframe which includes user's gender, age and like id data
Input:
    userDF: userDataStruct, generated by sampleInputPd
    likesize: minimum appearance of like id to use
Output:
    dataframe, feature table
'''
def likesDataprepocessing(userDF,likesize):
    
    print(likesize)
    likesData = pd.merge(userDF.userData,userDF.likesData,on='userid',how= 'right')
    likesData = likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > likesize)
    likesData = likesData.reindex()
    #print(likesData.index)
    userlikegroup = likesData.groupby('like_id').groups
    useridgroup = likesData.groupby('userid').groups

    newfeature = pd.DataFrame(columns=userlikegroup.keys())
    newfeature['userid']=userDF.userData['userid']
    newfeature['gender']=userDF.userData['gender']
    newfeature['age_group']=userDF.userData['age'].apply(groupAge)

    newfeature = newfeature.fillna(0)
    
    for key in useridgroup.keys():
        iuserid = key
        tempindex= useridgroup[iuserid]
        target = likesData.ix[tempindex,['like_id']]#.reindex()
        newfeature.ix[newfeature['userid'] == iuserid,target['like_id']]=1
    
    return newfeature

    
    
    
    