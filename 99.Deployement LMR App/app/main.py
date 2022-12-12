import os 
from fastapi import FastAPI
from fastapi import Form
import shutil
import pandas as pd 
import numpy as np 
import spacy
from fastapi.responses import FileResponse
from fastapi import Form
from fastapi import File
from fileinput import filename
from fastapi import UploadFile
from starlette.responses import HTMLResponse
from fastapi.responses import HTMLResponse
import warnings
from spacy import displacy
import json
import random
import pickle
import spacy
from spacy.tokens import DocBin
import warnings
import argparse
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_colwidth",2)

upload_path="app/geoai/input.jsonl"
model=spacy.load('./app/model-best')

def open_json_data(data_path):
    mode="r"
    dict_list = []
    with open(data_path,mode) as data_cleaned:
        for jobject in data_cleaned:
            jdict = json.loads(jobject)
            dict_list.append(jdict)
        # data_cleaned=json.load(data_cleaned)
    return dict_list

#function to compute the start and end offset of the location mention indentified
def find_index(sentence,word):
    str2=word
    str1=sentence
    start_index=str1.index(str2)
    length_word=len(str2)
    end_index=start_index+length_word
    start=start_index
    end=end_index
    return start,end

app=FastAPI()
@app.get("/")
async def main():
    content = """
<body>
<pre>
<p>
upload your file in this format please : 

{"tweet_id":"YOUR TWEET ID","text":"YOUR TWEET CONTAINS CRISS GOES HERE"}

</p>
</pre>
<form action="/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.post("/")
async def main(files: UploadFile=File(...)):
    if files.filename.endswith(".jsonl"):
            with open(upload_path,"wb") as buffer:
                shutil.copyfileobj(files.file,buffer)
            data=upload_path
            if os.path.getsize(data)==0:
                return {"Error":"The file is empty"}
            
            else:
                data_c=open_json_data(data)
                data = []
                tweet_id = []
                for d in data_c:
                    data.append(d['text'])
                    tweet_id.append(d['tweet_id'])
            
                # data=data_c['text']
                # tweet_id=data_c['tweet_id']
                
                if data and tweet_id:
                    # data_located_list=""
                    twitter_id=tweet_id
                    all_locations = []
                    # a=data_c['text']
                    # list_locations=[]
                    # dict_list = []
                    for data_item in data:
                        test_preds=model(data_item) #find recognition
                        list_locations = []
                        for entity in test_preds.ents:
                            if entity.label_=='Location':
                             list_locations.append(entity.text)
                        all_locations.append(list_locations)
                        
                    DICT_LIST = []
                    for a, locs in zip(data, all_locations):
                        dict_list = []
                        for b in locs:
                                data_located_Dict={}
                                data_located_Dict['text']=b
                                start_index, end_index = find_index(a, b) 
                                data_located_Dict['start_offset'] = start_index
                                data_located_Dict['end_offset'] = end_index
                                dict_list.append(data_located_Dict)
                        DICT_LIST.append(dict_list)
                        
                
                    # def find_index(sentence,word):
                    #     str2=word
                    #     str1=sentence
                    #     start_index=str1.index(str2)
                    #     length_word=len(str2)
                    #     end_index=start_index+length_word
                    #     start=start_index
                    #     end=end_index
                    #     return start,end
                    
                    # test_preds=model(data_c['text'])
                    # for entity in test_preds.ents:
                    #     if entity.label_=='Location':
                    #         list_locations.append(entity.text)
                    
                    # list_locs = set(list_locations)
                    # list_locations = set(list_locs)
                    
                    # for b in list_locations:
                    #     data_located_Dict={}
                    #     data_located_Dict['text']=b
                    #     start_index,end_index=find_index(a,b) 
                    #     data_located_Dict['start_offset']=start_index
                    #     data_located_Dict['end_offset']=end_index
                    #     dict_list.append(data_located_Dict)
                    
                    #Final Results
                    Model_Results = []
                    for tid, dicts in zip(tweet_id, DICT_LIST):
                         Model_Results.append({"tweet_id":tid,"location_mentions":dicts})
                    results = []
                    for item in Model_Results:
                        results.append(item.__str__())
                    with open("app/geoai/output.jsonl", "a") as f:
                        f.writelines(results)
                        f.close()
                    # outputs = open('app/geoai/output.jsonl','w')
                    # outputs.write(Model_Results.__str__())
                    # return Model_Results
                    return results
                
                else:
                    return {"Error":"no data or tweet id found in the file"}
    else:
        return {"Error":"Please upload a jsonl file only"}