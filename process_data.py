import os
import pandas as pd
import re
import sqlite3


current_folder = os.getcwd()
data_file = os.path.join(current_folder, 'data.tsv')

def clean_data(st):
    # clean accents and tags
    st = st.replace('â', 'a').replace('ç', 'c').replace('ё', 'e').replace('ï', 'i').replace('ô', 'o').replace('š','s').replace('û','u').replace('<br />', '')
    # remove repeating bad at the end
    bad_regex = r"( bad)*$"
    st = re.sub(bad_regex, '', st, re.MULTILINE | re.IGNORECASE)
    return st



def load_data(file):
    # with open(file) as f:
    #     print(f)
    df = pd.read_csv(file, header= None, delimiter='\t', error_bad_lines=False, encoding='utf-8', names=['review', 'sentiment'])
    df.iloc[:,0] = df.iloc[:,0].apply(lambda st: clean_data(st))
    df.iloc[:,1] = df.iloc[:,1].apply(lambda st: 1 if st == 'positive' else 0)
    return df

def store_data(df):
    conn = sqlite3.connect('./db.sqlite')
    df.to_sql('movie_reviews', conn, if_exists='replace', index=False)




data = load_data(data_file)
store_data(data)

# list(set(''.join(df_samples.iloc[:,0].values)))