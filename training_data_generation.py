import random
import math
#from flask.globals import request
from flask import Flask, render_template
import numpy as np
import torch
import pandas as pd
import io
import requests

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sqlalchemy import create_engine

import nltk
import json
import psycopg2

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keybert import KeyBERT

#engine = create_engine('postgeresql+psycopg2://signals:insecure@http://10.54.244.63:5432/signals')
#df.to_sql('cleaned_table', engine, if_exists='replace',index=False)

#####################################################################################################

DATA_PATH = lambda: '/training/Training_data_generation/files'
CLEAN_FILE = 'cleaned_file.csv'
CLEAN_PATH = lambda: f'{(DATA_PATH())}/{CLEAN_FILE}'
GENERATED_FILE = 'generated_data.csv'
GENERATED_PATH = lambda: f'{(DATA_PATH())}/{GENERATED_FILE}'
CAT_FILE = 'category_keyword.csv'
FREQ_FILE = 'keyword_frequency.csv'

#####################################################################################################

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

#####################################################################################################

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

#####################################################################################################  

def similarity_fun(X, Y):
  #tokenization
  X_list = word_tokenize(X)
  Y_list = word_tokenize(Y)

  #list the stopwords
  sw = stopwords.words('english')
  l1 = []; l2 = []

  #remove stopwords from texts
  X_set = {w for w in X_list if not w in sw}
  Y_set = {w for w in Y_list if not w in sw}

  #set containing keywords from both texts
  rvector = X_set.union(Y_set)
  for w in rvector:
    #create vectors
    if w in X_set: l1.append(1)
    else: l1.append(0)
    if w in Y_set: l2.append(1)
    else: l2.append(0)
  c = 0

  #cosine similarity
  for i in range(len(rvector)):
    c = c + l1[i]*l2[i]
  cosine = c / float((sum(l1)*sum(l2))**0.5)

  return cosine

##################################################################################

def input_training_data(trainingdata):
    path = f'{DATA_PATH()}/{trainingdata}'
    sheetname = 'English'
    df = pd.read_excel(path, sheet_name= sheetname)

    #Count Total Rows
    count_rows = df.shape[0]

    #Count NULL Rows
    null_count = df['description_message'].isnull().values.sum()

    #Count Duplicate Sentences
    count_duplicates = 0
    duplicates = df.pivot_table(index=['description_message'], aggfunc='size')
    for duplicate in duplicates:
        if duplicate > 1:
            count_duplicates = count_duplicates + duplicate

    #Count Rows with 2 or less word
    data = df[df.description_message.str.count(' ') < 3]
    count_2words = len(data.description_message)

    #Dataframe sample
    data_sample = df.head()
    data_sample = data_sample.to_dict()

    conn = 'postgresql://signals:insecure@localhost:5432/signals'
    engine = create_engine(conn)
    db = engine.connect()

    df.to_sql('input_table', con=db, if_exists='replace',index=False)

    #df_db = pd.read_sql_query('select * from "cleaned_table"', con = engine)
    #print(df_db.head())

    db.close()

    original_summary = {
    'Total Rows' : int(count_rows),
    'NULL Rows' :  int(null_count),
    "Duplicate Sentences" : int(count_duplicates),
    "Description with 2 or less words" : int(count_2words),
    "First few rows of input sheet" : data_sample
    }

    return json.dumps(original_summary)

#########################################################################################

def clean_data():
    #sheetname = 'English'
    #path = f'{DATA_PATH()}/{trainingdata}'

    conn = 'postgresql://signals:insecure@localhost:5432/signals'
    engine = create_engine(conn)
    db = engine.connect()

    #df = pd.read_excel(path, sheet_name= sheetname)
    df = pd.read_sql_query('select * from "input_table"', con = engine)

    #Remove Rows With Duplicate descriptions
    df = df.drop_duplicates(subset=['description_message'])
    #Remove Rows With 2 or less words and Null values
    df = df[df.description_message.str.count(' ') > 2]
    #Reset Index
    df.reset_index(drop=True, inplace=True)

    col_names =  ['Description', 'Category', 'SubCategory1', 'SubCategory2', 'SubCategory3']

    df_new = pd.DataFrame(columns = col_names)
    df_new['Description'] = df['description_message']
    df_new['Category'] = df['cat1']
    df_new['SubCategory1'] = df['cat2']
    df_new['SubCategory2'] = df['cat3']
    df_new['SubCategory3'] = df['cat4']


    #Extract keywords from sentences and save it as per categories
    kw_model = KeyBERT()
    df_cat = df_new.groupby('SubCategory2').agg({'Description':lambda x: list(x)})
    categories = df_cat.index

    cats = []
    for k in range(len(categories)):
      f = categories[k]
      cats.append(f)

    cat_dict = {}
    freq_dict = {}
    i = 0 

    for l in df_cat['Description']:
      text = ' '.join(map(str,l))
      s = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1))
      cat_dict[cats[i]] = s
      j = 0
      key_freq = []
      for m in s:
        frequency = text.count(m[0])
        output_key_freq = (m[0], frequency)
        key_freq.append(output_key_freq)
        j = j + 1
      freq_dict[cats[i]] = key_freq
      i = i + 1


    df_cat_des = pd.DataFrame.from_dict(cat_dict, orient='index').transpose()
    #print(df_cat_des.head())

    df_cat_freq = pd.DataFrame.from_dict(freq_dict, orient='index').transpose()
    #print(df_cat_freq.head())

    #Save cleaned data
    df_new.to_csv(f'{DATA_PATH()}/{CLEAN_FILE}')

    #Save category wise description keywords
    df_cat_des.to_csv(f'{DATA_PATH()}/{CAT_FILE}')

    #Save category wise keywords frequency
    df_cat_freq.to_csv(f'{DATA_PATH()}/{FREQ_FILE}')

    #Get first few rows of cleaned data as dictionary
    data = df_new.head()
    data = data.to_dict()

    #Get first few rows of category keyword frequency dataframe as dictionary
    data_cat = df_cat_des.head()
    data_cat = data_cat.to_dict()

    #Get first few rows of category keyword dataframe as dictionary
    data_freq = df_cat_freq.head()
    data_freq = data_freq.to_dict()

    #Count Total Rows
    count = df_new['Description'].describe()
    count_rows = count[0]

    #Count category wise sentences
    count_categories = df_new.groupby('SubCategory2').count()[['Description']]
    count_categories = count_categories.to_dict()

    summary = {
        'Rows In Cleaned Sheet' : int(count_rows),
        'Cleaned Sheet Sample' : data,
        'Category Keyword Sample' : data_cat,
        'Category Keyword Frequency' : data_freq,
        'Category Wise Sentences' : count_categories
    }

    
    df_new.to_sql('cleaned_table', con=db, if_exists='replace',index=False)

    df_db = pd.read_sql_query('select * from "cleaned_table"', con = engine)
    print(df_db.head())

    db.close()

    return json.dumps(summary)

#########################################################################################
def send_request_no(request_no):
  global request
  request = request_no

def new_training_data():
    num_beams = 10
    num_return_sequences = 5

    request_no = request

    #df = pd.read_csv(f'{DATA_PATH()}/{CLEAN_FILE}')
    
    conn = 'postgresql://signals:insecure@localhost:5432/signals'
    engine = create_engine(conn)
    db = engine.connect()

    df = pd.read_sql_query('select * from "cleaned_table"', con = engine)
    print(df.head())

    df_new = pd.DataFrame(columns=['Description', 'Category', 'SubCategory1', 'SubCategory2', 'SubCategory3'])
    descriptions = []
    categories = []
    subcat1 = []
    subcat2 = []
    subcat3 = []

    for l in range(len(df)):
      descriptions.append(df.loc[l, 'Description'])
      categories.append(df.loc[l, 'Category'])
      subcat1.append(df.loc[l, 'SubCategory1'])
      subcat2.append(df.loc[l, 'SubCategory1'])
      subcat3.append(df.loc[l, 'SubCategory1'])

    #Declare variables to store new sentences
    d = []
    c = []
    s1 = []
    s2 = []
    s3 = []

    row = 0

    # Generate new data and remove any sentences with similarity score greater than 0.8
    for text in descriptions:
      data = str(text)
      print(row, data)
      sentences = get_response(data,num_return_sequences,num_beams)
      texts = sentences
      i = 0
      flag = 1
      for sentence in sentences:
        similarity = similarity_fun(data, sentence)
        if 0.8 > similarity:
          j = 0
          for text in texts:
            if i != j:
              similarity = similarity_fun(sentence, text)
              if 0.9 > similarity:
                flag = 1
              else:
                flag = 0
                break
            else:
              continue
            j = j + 1
          if flag == 1:
            d.append(sentence)
            c.append(categories[row])
            s1.append(subcat1[row])
            s2.append(subcat2[row])
            s3.append(subcat3[row])
        i = i + 1
      
      progress(request_no ,row, len(descriptions))
      row = row + 1

    df_new['Description'] = d
    df_new['Category'] = c
    df_new['SubCategory1'] = s1
    df_new['SubCategory2'] = s2
    df_new['SubCategory3'] = s3

    df_new.to_csv(f'{DATA_PATH()}/{GENERATED_FILE}')

    table_name = f'generated_table_{request_no}'

    df_new.to_sql('table_name', con=db, if_exists='replace',index=False)
    df_db = pd.read_sql_query('select * from "table_name"', con = engine)
    print(df_db.head())
    db.close()

    data = df_new.head()
    data = data.to_dict()

    #Count Total Rows
    count = df_new['Description'].describe()
    count_rows = count[0]

    summary = {
        'Rows In Generated Sheet' : int(count_rows),
        'Generated Sheet Sample' : data
    }

    #return json.dumps(summary)


progress_global = {} 

def progress(request_no ,row, rows):
  pro = row*100/rows
  if (pro < 0.95):
    pro = 0
  if (pro % 1 > 0.95 or pro == 0):
    pro = math.ceil(pro)
    #request_text = str(request_no)
    progress_global[str(request_no)] = pro
    print('\n',pro,'\n')

def progress_api(request_no):
  if str(request_no) in progress_global.keys():
    progress = progress_global[str(request_no)]
    #progress_text = f'{str(request_no)} : {progress}'
    progress_text = {
      'Request Number' : request_no,
      'Progress' : progress
    }
    return json.dumps(progress_text)
  else:
    return json.dumps('Request not found')


def generated_data(request_no):
  conn = 'postgresql://signals:insecure@localhost:5432/signals'
  engine = create_engine(conn)
  db = engine.connect()

  del progress_global[str(request_no)]

  table_name = f'generated_table_{request_no}'

  df = pd.read_sql_query('select * from "table_name"', con = engine)
  db.close()

  data = df.head()
  data = data.to_dict()

  #Count Total Rows
  count = df['Description'].describe()
  count_rows = count[0]

  summary = {
      'Rows In Generated Sheet' : int(count_rows),
      'Generated Sheet Sample' : data
  }

  return json.dumps(summary)

