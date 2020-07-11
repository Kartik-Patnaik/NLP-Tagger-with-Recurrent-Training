import os 
from nltk.tag import StanfordNERTagger  
import json
from flask import Flask,request 
import nltk
import pandas as pd


java_path = "java.exe"
os.environ['JAVAHOME'] = java_path

app = Flask(__name__)
@app.route('/nlp/processSentence')  



def recognizer():
    exists = os.path.isfile('C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/corpus-tagging.ser.gz')
    if exists:
        jar = 'C:/Users/KPATNAIk/Desktop/Topic_Modelling1/stanford-ner.jar'
        model = 'C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/corpus-tagging.ser.gz'
#        sentence = "there was very limited stock on my bra size but the customer staff was good"
        sentence = request.args.get('sentence')
        sentencex = "."
        sentence = sentence+sentencex
        sentence = sentence.lower()
        ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
        words = nltk.word_tokenize(sentence)
        results = ner_tagger.tag(words)
        filter = ['o']
        ls2 = [(x,y) for (x,y) in results if y not in filter] 
    
        d = {}
        for a, b in ls2:
            d.setdefault(b, []).append(a)
        new_abc = [ [ ' '.join(d.pop(b)), b ] for a, b in ls2 if b in d ]
        sub_dict = json.dumps(new_abc)
        
        if new_abc!=[]:
            new_abc = pd.DataFrame(new_abc)
            new_abc.columns = ['sentence', 'output']
            new_abc = new_abc[new_abc.output != 'O']
            new_abc = new_abc.set_index('output')['sentence'].to_dict()
            sub_dict = json.dumps(new_abc)
            df = pd.DataFrame({'input':[sentence]})
            df["output"] = [ls2]
            exists = os.path.isfile("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/user_entered_sc.csv")
            if exists:
                df1=pd.read_csv("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/user_entered_sc.csv")
                df1 = df1.append(df) 
                df1['New_ID'] = range(1, 1+len(df1))
                df1.to_csv("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/user_entered_sc.csv", sep=',',index=False)
    
            else:
       
                df.to_csv("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/Test_Topic1/user_entered_sc.csv", sep=',',index=False)
        else:
            sub_dict = {"blank":"blank"}
            sub_dict = json.dumps(d) 
            
    else:
        d = {}
        sub_dict = {"blank":"blank"}
        sub_dict = json.dumps(d)
    
    return sub_dict

if __name__ == '__main__': 
    app.debug = True
    app.run(host = '0.0.0.0',port = 1313)


#ls2 = [('buy', 'product'), ('from', 'o'), ('sachin', 'trader_name')]
#sub_dict= {'output':ls2}
#sub_dict = json.dumps(sub_dict)
