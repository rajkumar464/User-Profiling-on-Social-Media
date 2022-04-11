#!/bin/bash
# author: Sajad H Alhamada
# Text Model: using both LIWC.csv and text folder. 
# To predect: five personality triats and gender.
# The below code accomedate 'testing' phase only. It loads 12 .pkl files as follow:
#['tfid_vect.pkl','text_model.pkl',
# 'support_e.pkl','text_model_e.pkl',
# 'support_n.pkl','text_model_n.pkl',
# 'support_a.pkl','text_model_a.pkl',
# 'support_o.pkl','text_model_o.pkl',
# 'support_c.pkl','text_model_c.pkl']
# if the above files did not exist, please run 'text_model.py' and 'text_model_v2.py' to generate them.
# You might get UserWarning if you run 'text_model_v2.py', if so please ignore it. It is sklearn bug that does NOT effect the results.  
# Expected Results: gender:0.7515 ext:0.65 neu:0.60 agr:0.42 con: 0.48 ope:0.38

import glob,os,sys, getopt
from xml.dom import minidom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
import pandas as pd
from nltk.util import ngrams
import joblib,re

def build_ngrams(text, n=1):
    tokens = re.findall(r"\w+", str(text.lower().split()))
    return list(ngrams(tokens, n))

def find_file(filename, path):
    for root,dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root,filename)

def find_dir(dire,path):
    for root,dirs, files in os.walk(path):
        if dire in dirs:
            return os.path.join(root,dire)

def test(input_path,out_path):

    # load text files for gender prediction
    path= find_dir('text',input_path)
    files= [file for file in glob.glob(str(path)+'/*.txt')]
    df_test_x = pd.DataFrame()
    for file in files:
        with open( file,'r', encoding='cp1252') as fr:
            line = fr.read()
            row={'text':str(line),'userId':str(file)[:-4].replace(path+'/','')}
            df_test_x = df_test_x.append(row,ignore_index=True)

    tfid_vect = joblib.load('tfid_vect.pkl')
    x_test= tfid_vect.transform(df_test_x['text'])

    #load LIWC.csv for Personality traits predections
    df_test_x_p= pd.read_csv(find_file('LIWC.csv',input_path))
    ids = df_test_x_p['userId']
    df_test_x_p= df_test_x_p.drop(['Seg','userId'],axis=1)

    # predectiong Gneder
    lr = joblib.load('text_model.pkl')
    lr_pred_g= lr.predict(x_test)

    # predectiong extrovert
    support = joblib.load('support_e.pkl')
    lr = joblib.load('text_model_e.pkl')
    lr_pred_e = lr.predict(df_test_x_p.loc[:,support])

    # predectiong neurotic
    support = joblib.load('support_n.pkl')
    lr = joblib.load('text_model_n.pkl')
    lr_pred_n = lr.predict(df_test_x_p.loc[:,support])

    # predectiong agreeable
    support = joblib.load('support_a.pkl')
    lr = joblib.load('text_model_a.pkl')
    lr_pred_a = lr.predict(df_test_x_p.loc[:,support])

    # predectiong conscientious
    support = joblib.load('support_c.pkl')
    lr = joblib.load('text_model_c.pkl')
    lr_pred_c = lr.predict(df_test_x_p.loc[:,support])

    # predectiong open
    support = joblib.load('support_o.pkl')
    lr = joblib.load('text_model_o.pkl')
    lr_pred_o = lr.predict(df_test_x_p.loc[:,support])


    for row in range(len(ids)):
        
        gnd= ('male' if lr_pred_g[row]==0 else 'female')
        ext= lr_pred_e[row]
        neu= lr_pred_n[row]
        agr= lr_pred_a[row]
        con= lr_pred_c[row]
        ope= lr_pred_o[row]

        data = {'id':ids[row],'gnd':gnd,'ext':ext,'neu':neu,'agr':agr,'con':con,'ope':ope}
        generate_xml(out_path,data)
    

def generate_xml(out_path,data):
    root = minidom.Document()

    user= root.createElement('user')
    user.setAttribute('id',data['id'])
    user.setAttribute('age_group','xx-24')
    user.setAttribute('gender',data['gnd'])
    user.setAttribute('extrovert',str(data['ext']))
    user.setAttribute('neurotic',str(data['neu']))
    user.setAttribute('agreeable',str(data['agr']))
    user.setAttribute('conscientious',str(data['con']))
    user.setAttribute('open',str(data['ope']))
    root.appendChild(user)

    xml_str= root.toprettyxml(indent='\t')

    out_path = ( out_path+'/' if out_path[len(out_path)-1] != '/' else out_path)
    save_path = out_path + data['id']+'.xml'

    with open(save_path,'w') as file:
        file.write(xml_str)

def makdir_output(out_path):
    isExist = os.path.exists(out_path)
    if not isExist:
        os.makedirs(out_path)

def main():
    input_path = ''
    output_path   = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:o:",[])
    except getopt.GetoptError:
        print('-i <inputDirectory> -o <outputDirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-i'):
            input_path = arg
        elif opt in ('-o'):
            output_path = arg
    
    makdir_output(output_path)
    test(input_path,output_path)
    print('done!')

if __name__== "__main__":
    main()
