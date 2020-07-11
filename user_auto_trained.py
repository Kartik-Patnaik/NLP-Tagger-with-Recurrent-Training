import os
import pandas as pd
from flask import Flask,request 
import pathlib
from flask_cors import CORS, cross_origin
import requests
import re
import nltk
import datefinder

java_path = "java.exe"
os.environ['JAVAHOME'] = java_path
#os.chdir("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train2/")
app = Flask(__name__) 
CORS(app, support_credentials=True)
cwd = os.getcwd()
@app.route('/nlp/tags',methods=['GET', 'POST'] ) 
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
    response = requests.get(vf_url,headers={"Authorization":Authorization})#    df1=pd.read_csv("C:/Users/kartik.patnaik/Desktop/mobileapp/new_test/stanford-ner-2018-10-16/train2/Book1.csv")
    if response.status_code == 200:
        ROOT_PATH = os.getenv("path_root_url")
        os.chdir(ROOT_PATH)
        df2 = str(tenant_id)+"/"+str(appid)+"/"+str(object_id)
        user_input = request.get_json()
        if user_input != {}:
            wanted_keys = ['sentence']
            wanted_keys1 = ['Tags']
            sentence = {k: user_input[k] for k in set(wanted_keys) & set(user_input.keys())}
            sentence = list( sentence.values() )[0]
            if sentence != None and sentence != '':
                sentence = sentence.lower()               
                article = sentence[:]
                def find_match(sentence,df):
                    for i in range(df.shape[0]):
                        if sentence.find(df['rpl'][i]) !=-1:
                            sentence = sentence[:sentence.find(df['rpl'][i])] +  df['rpl1'][i] +  sentence[sentence.find(df['rpl'][i])+ len(df['rpl'][i]):]
                    return sentence
                ls3 = list(datefinder.find_dates(sentence, source=True))

                if ls3!=[]:
                    ls4 = pd.DataFrame(ls3)
                    ls4.columns = ["rpl1","rpl"]
                    ls4["rpl1"] = ls4["rpl1"].dt.strftime('%Y-%m-%d')
                    sentence = find_match(article,ls4)

                tags = {k: user_input[k] for k in set(wanted_keys1) & set(user_input.keys())}
                tags = list( tags.values() )[0]
                def lower_dict(d):
                    new_dict = dict((k, v.lower()) for k, v in d.items())
                    return new_dict
                tags = lower_dict(tags)
                new_list = [] 
                for key, value in tags.items():
                    new_list.append([key, value])
                ui1 = pd.DataFrame(new_list)
                ui1.columns = ['action','sentence']
                #ui2 = ui1.sentence.str.split(expand=True,)
                ui1[['sentence1','sentence2']] = ui1['sentence'].str.split(' ', n=1, expand=True)
                ui2 = ui1[['sentence1','action']]
                ui3 = ui1[['sentence2','action']]
                #ui3.dropna(subset=['action'],inplace = True) 
                ui3.dropna(inplace = True)    
                ui2.columns = ['sentence', 'action']
                ui3.columns = ['sentence', 'action']
                ui4 = ui2.append(ui3, ignore_index=True)            
                lst_ip1 = nltk.word_tokenize(sentence)
                lst_ip3 = pd.DataFrame(lst_ip1)
                lst_ip3.columns = ['sentence']
                
                #################################################join
                result = pd.merge(lst_ip3,
                                 ui4,
                                 on='sentence', 
                                 how='left')
                
                result['action'] = result['action'].fillna('o')
                result['sentence'] = result['sentence'].map(str) + " " + result["action"]
                user_input3 = result['sentence']
                user_input3.to_csv(str(df2) +'/user_input3.tsv',header=False, index=False)
                user_input3 = pd.read_csv(str(df2) +'/user_input3.tsv', sep='\t',header = None)
                exists = os.path.isfile(str(df2) +'/dummy-corpus1.tsv')
                exists1 = os.path.isfile(str(df2) +'/dummy-corpus2.tsv')
                if exists and not exists1:
                    pa1 = pd.read_csv(str(df2) +'/dummy-corpus1.tsv', sep='\t',header = None)
                    pa2 = pa1.append(user_input3,ignore_index=True)
                    pa2 = pa2.append([". o"])
                elif exists1 and exists:
                    pa1 = pd.read_csv(str(df2) +'/dummy-corpus2.tsv', sep='\t',header = None)
                    pa2 = pa1.append(user_input3,ignore_index=True)
                    pa2 = pa2.append([". o"])  
                else:
                    pa2 = user_input3
                    pa2 = pa2.append([". o"])
                    
                pa2.to_csv(str(df2) +'/dummy-corpus2.tsv',header=False, index=False)
                cwd = os.getcwd()
                cwd = pathlib.PureWindowsPath(cwd)
                cwd = cwd.as_posix()
                prop = "trainFile = "+ str(cwd) +"/" + str(df2) + """/dummy-corpus2.tsv
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
                
                file = open( str(cwd) +"/" + str(df2)+'/prop2.txt', 'w')
                file.write(prop)
                file.close()
                myCmd = 'java -jar stanford-ner.jar -mx4g -prop' " "  + str(df2) + '/prop2.txt'
                os.system(myCmd)   
    
                return 'Recurrent Training on Completed Successfully'
            else:
                return 'No Data to be trained on NULL'
    else:
        return 'Unsuccessful Auth'
    
if __name__ == '__main__': 
    app.debug = True
    app.run(host = '0.0.0.0',port = 1414)
