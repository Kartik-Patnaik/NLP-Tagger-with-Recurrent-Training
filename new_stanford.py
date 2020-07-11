import os 
import re
from nltk.tag import StanfordNERTagger  
import json
from flask import Flask,request 
import nltk
import pandas as pd
from flask_cors import CORS, cross_origin
import pathlib
import requests
import datefinder

#Finding the os directory and corret java path
#os.chdir("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train1")
#java_path = "C:/Program Files/Java/jre1.8.0_191/bin/java.exe" 
#os.environ['JAVAHOME'] = java_path


#myCmd = 'java -jar C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/stanford-ner.jar -mx4g -prop C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train/prop1.txt'
#os.system(myCmd)

#Flask code
#os.chdir("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train2/")
java_path = "java.exe"
os.environ['JAVAHOME'] = java_path
app = Flask(__name__)
CORS(app, support_credentials=True)
@app.route('/nlp/processSentence')  
@cross_origin(supports_credentials=True)

def recognizer():
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
    if response.status_code == 200:
        ROOT_PATH = os.getenv("path_root_url")
        os.chdir(ROOT_PATH)
        df2 = str(tenant_id)+"/"+str(appid)+"/"+str(object_id)
        cwd = os.getcwd()
        cwd = pathlib.PureWindowsPath(cwd)
        cwd = cwd.as_posix()
        if not os.path.exists(str(cwd) +"/" + str(df2)):
            os.makedirs(str(cwd) +"/" + str(df2))
        
        exists = os.path.isfile(str(df2) + '/corpus-tagging.ser.gz')
        if exists:
            jar = 'stanford-ner.jar'
            model = str(df2) + "/corpus-tagging.ser.gz"
            sentence = request.args.get('sentence')
#            sentence = sentence.lower()            
            if sentence !="":
                article = sentence[:]
                def find_match(sentence,df):
                    for i in range(df.shape[0]):
                        if sentence.find(df['rpl'][i]) !=-1:
                            sentence = sentence[:sentence.find(df['rpl'][i])] +  df['rpl1'][i] +  sentence[sentence.find(df['rpl'][i])+ len(df['rpl'][i]):]
                    return sentence
                ls3 = list(datefinder.find_dates(sentence, source=True))
#                if ls3!=[]:
#                    ls4 = pd.DataFrame(ls3)
#                    ls4.columns = ["rpl1","rpl"]
#                    ls4["rpl1"] = ls4["rpl1"].dt.strftime('%Y-%m-%d')
#                    sentence = find_match(article,ls4)
                
                sentence1 = sentence.lower()
                ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
                words = nltk.word_tokenize(sentence1)
                results = ner_tagger.tag(words)
                filter = ['T']
                ls2 = [(x,y) for (x,y) in results if y not in filter] 
                df = pd.DataFrame(ls2)  
                lst_ip1 = nltk.word_tokenize(sentence)
                lst_ip3 = pd.DataFrame(lst_ip1)
                df[0]=lst_ip3[0]                               
                ls2 = list(df.itertuples(index=False, name=None))
                
                d = {}
                for a, b in ls2:
                    d.setdefault(b, []).append(a)
                new_abc = [ [ ' '.join(d.pop(b)), b ] for a, b in ls2 if b in d ]
                sub_dict = json.dumps(new_abc)
                
                if new_abc!=[]:
                    new_abc = pd.DataFrame(new_abc)
                    new_abc.columns = ['sentence', 'output']
                    new_abc = new_abc[new_abc.output != 'T']
                    new_abc = new_abc.set_index('output')['sentence'].to_dict()
                    sub_dict = json.dumps(new_abc)
                    df = pd.DataFrame({'input':[sentence]})
                    df["output"] = [ls2]
                    exists = os.path.isfile(str(df2) + "/user_entered.csv")
                    if exists:
                        df1=pd.read_csv(str(df2) + "/user_entered.csv")
                        df1 = df1.append(df) 
                        df1['New_ID'] = range(1, 1+len(df1))
                        df1.to_csv(str(df2) + "/user_entered.csv", sep=',',index=False)
            
                    else:
               
                        df.to_csv(str(df2) + "/user_entered.csv", sep=',',index=False)
                else:
                    sub_dict = {"blank":"blank"}
                    sub_dict = json.dumps(d)
            else:
                d = {}
                sub_dict = {"blank":"blank"}
                sub_dict = json.dumps(d)
                
        else:
            d = {}
            sub_dict = {"blank":"blank"}
            sub_dict = json.dumps(d)
            
        return sub_dict
    else:
        return 'Unsuccessful Auth'

if __name__ == '__main__': 
    app.debug = True
    app.run(host = '0.0.0.0',port = 1313)


#ls2 = [('buy', 'product'), ('from', 'o'), ('sachin', 'trader_name')]
#sub_dict= {'output':ls2}
#sub_dict = json.dumps(sub_dict)
