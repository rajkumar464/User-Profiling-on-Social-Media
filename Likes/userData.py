'''
TCSS 555 Winter 2022 Team Project
Struct for storing user's data
Author: Zhengwu Liu
'''

'''
Struct of each user's data, detail of usage in inputTools.py, main.py
id: string, userid
likeID: string list, include user's liked id.
age: float, user's age
age_group: string, group of age, includes: "xx-24", "25-34", "35-49", "49-xx"
gender: string, gender
genderType: float, 0 for male, 1 for female
ext,neu,agg,con,ope: float, LIWC data
'''


class user(object):
    idList = []

    id = ""
    statusUpdates = []
    likeID = []
    LIWC = [-1 for x in range(0, 81)]

    age = -1
    age_group = "xx-24"
    gender = "male"
    genderType = 1

    extrovert = 1
    neurotic = 1
    agreeable = 1
    conscientious = 1
    open = 1

    def __init__(self, params):
        '''
        Constructor
        '''
