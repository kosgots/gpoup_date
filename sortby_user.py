# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:45:06 2020

@author: Kostas Gotsoulias
"""

import feather
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np 
import pandas as pd
from sklearn import model_selection
from pandas.util.testing import assert_frame_equal
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from string import Template
from datetime import datetime
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
import gensim
import time

conv = feather.read_dataframe("C:/Users/Kostas/.spyder-py3/chats.feather")

df1=conv

df1 = [x for _, x in df1.groupby(conv.from_user)]



#elegxos an exei ginei sosto xorisma kata chat
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()



y1=False
for i in  range(0, len(df1)):
    y1=is_unique(df1[i].from_user)
    if not y1:
        print(y1)
        
        
        
# my_template = Template("C:/Users/Kostas/Desktop/temp/csv/userdate${x}.csv")
# for x in range(0,10000):
#     my_template.substitute(x=x)

for x in range(0,1): 
# for x in range(0,len(df1)):  
    df1[x]['datetime'] = pd.to_datetime(df1[x]['timestamp'], unit='ms')
    df1[x]['datetime'] = pd.to_datetime(df1[x]['timestamp'], unit='ms')
    df1[x]['month'] =  df1[x]['datetime'].dt.month_name()
    df1[x]['day'] =  df1[x]['datetime'].dt.day_name()
    df1[x]['hour']= df1[x]['datetime'].dt.hour
    df1[x]['date']= df1[x]['datetime'].dt.date
  

# save df1
# Akolouthoun templates gia automatopoimeni apothikeush arxeiwn
# p=list(range(0, len(df1)))
p1=list(range(0,  len(df1)))

my_template1 = Template("C:/Users/Kostas/Desktop/temp/csv_user/user${id1}.csv")
for id1 in p1:
    my_template1.substitute(id1=id1)

for id1 in p1:
    df1[id1].to_csv(open(my_template1.substitute(id1=id1), 'w', encoding="utf-8")) 
    
    
# load df1 (prospathw toylaxiston) 
# start = time.time()
# print("hello")
# end = time.time()
# print(end - start)

# df2=[] * 1323046
df2=[pd.DataFrame()] * 1323046

p2=list(range(0, 1323046))
my_template2 = Template("C:/Users/Kostas/Desktop/temp/csv_user/user${id2}.csv")

for id2 in p2:
    my_template2.substitute(id2=id2)

for id2 in p2:
       data=pd.read_csv(my_template2.substitute(id2=id2), "'w'",engine= 'python' , encoding="utf-8",names=['chat', 'from_user', 'timestamp', 'message', 'datetime', 'month', 'day', 'hour', 'date'])
       df2[id2]=df2[id2].append(data)


 
   
#dilwsi dataframe opou ths periexei sto 'watched'  ta metadata gia ton xrono
train_watched = pd.DataFrame(columns=['user_id', 'watched'])    

for i in p2:    
    for index, from_user in df1[i].from_user.iteritems():    
        time_user = " ".join([" ".join(map(str,  df1[i].chat)) +" ".join(map(str,  df1[i].datetime))])
        train_watched.loc[index] = [from_user, time_user]

# save train_watched dataframe
train_watched.to_csv(open('C:/Users/Kostas/Desktop/temp/trained.csv', 'w', encoding="utf-8"))


# load train_watched dataframe
#out of memory  
#train_watched=pd.read_csv('C:/Users/Kostas/Desktop/temp/trained.csv')  

# mylist = []

# for chunk in  pd.read_csv('C:/Users/Kostas/Desktop/temp/trained.csv', sep=';',names=['user_id','watched'], chunksize=20000):
#     mylist.append(chunk)

# train_watched = pd.concat(mylist, axis= 0)
# del mylist




list_doc = []

for row in train_watched.to_dict(orient='record'):
    list_doc.append(str(row['watched']).strip().split(' '))



model = Word2Vec(list_doc, window=5, min_count=1, workers=8)


def most_similar(aa):
    try:
        print("Similar of "+df1[df1['datetime'] == int(aa)].iloc[0]['from_user'])
    except:
        print("Similar of "+aa)
    return [(x, df1[df1['datetime'] == int(x[0])].iloc[0]['from_user']) for x in model.wv.most_similar(aa)]


most_similar('2017-10-10')

# save word2vec model
model.wv.save_word2vec_format('model.bin',binary=True)

# load word2vec model
model1 = gensim.models.Word2Vec.load("C:/Users/Kostas/Desktop/temp/model.bin")
