import os
from nltk.tag import StanfordNERTagger  
import re
import pandas as pd
import json
from flask import Flask,request 
import requests
from difflib import get_close_matches
import nltk
import pathlib
from flask_cors import CORS, cross_origin

java_path = "java.exe"
os.environ['JAVAHOME'] = java_path
#os.chdir("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train2/")
app = Flask(__name__) 
CORS(app, support_credentials=True)
@app.route('/raw')
@cross_origin(supports_credentials=True)

def upload():
    headers_post = request.headers 
    appid = headers_post['appId']
    tenant_id = headers_post['X-TenantID']
    object_id = headers_post['X-Object']
    Authorization = headers_post['Authorization'] 
    
#    dct_prop = dict(line.strip().split('=') for line in open('properties.txt'))
    URL = os.getenv("properties_url")
    response1 = requests.get(URL,headers={"Content-Type":"application/json","X-TenantID" : tenant_id})
    ENV_URL=response1.json()
    ENV_URL=ENV_URL["propertyValue"]
    vf_url = str(ENV_URL) +"/cac-security/api/userinfo"
    response = requests.get(vf_url,headers={"Authorization":Authorization})
#    df1=pd.read_csv("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train2/Book1.csv")
#    df2 = df1[(df1['appid']== appid) & (df1['tenant_id']== tenant_id) & (df1['object_id']== object_id)]['path'].values[0]
    if response.status_code == 200: 
        ROOT_PATH = os.getenv("path_root_url")
#        response_url = requests.get(ROOT_URL,headers={"Authorization":Authorization,"X-TenantID":tenant_id})
#        product=response_url.json()
#        root_product_url = product["propertyValue"]
        os.chdir(ROOT_PATH)
        df2 = str(tenant_id)+"/"+str(appid)+"/"+str(object_id)
        cwd = os.getcwd()
        cwd = pathlib.PureWindowsPath(cwd)
        cwd = cwd.as_posix()
        if not os.path.exists(str(cwd) +"/" + str(df2)):
            os.makedirs(str(cwd) +"/" + str(df2))
            pa2 = pd.DataFrame(index=range(1))
            pa2.to_csv(str(cwd) +"/" +str(df2) +'/dummy-corpus1.tsv',header=False, index=False)
            prop = "trainFile = "+ str(cwd) +"/" + str(df2) + """/dummy-corpus1.tsv
            serializeTo ="""+ str(cwd) +"/" + str(df2) +"""/corpus-tagging.ser.gz
            map = word=0,answer=1
            
            useClassFeature=true
            useWord=true
            useNGrams=true
            noMidNGrams=true
            maxNGramLeng=6
            usePrev=true
            useNext=true
            useSequences=true
            usePrevSequences=true
            maxLeft=1
            useTypeSeqs=true
            useTypeSeqs2=true
            useTypeySequences=true
            wordShape=chris2useLC
            useDisjunctive=true"""
            
            file = open( str(cwd) +"/" + str(df2)+'/prop1.txt', 'w')
            file.write(prop)
            file.close()
            myCmd = 'java -jar stanford-ner.jar -mx4g -prop' " "  + str(df2) + '/prop1.txt'
            os.system(myCmd)
        else:         
            prop = "trainFile = "+ str(cwd) +"/" + str(df2) + """/dummy-corpus1.tsv
            serializeTo ="""+ str(cwd) +"/" + str(df2) +"""/corpus-tagging.ser.gz
            map = word=0,answer=1
            
            useClassFeature=true
            useWord=true
            useNGrams=true
            noMidNGrams=true
            maxNGramLeng=6
            usePrev=true
            useNext=true
            useSequences=true
            usePrevSequences=true
            maxLeft=1
            useTypeSeqs=true
            useTypeSeqs2=true
            useTypeySequences=true
            wordShape=chris2useLC
            useDisjunctive=true"""
            
            file = open( str(cwd) +"/" + str(df2)+'/prop1.txt', 'w')
            file.write(prop)
            file.close()
            myCmd = 'java -jar stanford-ner.jar -mx4g -prop' " "  + str(df2) + '/prop1.txt'
            os.system(myCmd)  
        if os.system(myCmd)==0:
            return 'Raw Training on blank Completed Successfully'
        else:
            
            return 'Recurrent training Failed check Header or existance of input File'        
    else:
        return 'Unsuccessful Auth'
    
if __name__ == '__main__': 
    app.debug = True
    app.run(host = '0.0.0.0',port = 5000)