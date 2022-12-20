from dis import dis
from tkinter import Y
from contrastive_keyword_extraction import contrastive_extraction, final_score, combine_keywords
import sqlite3
import pandas as pd
from tqdm import trange
import string
import sentence_comparision
import sentence_importance
import streamlit as st
from baselines import create_inter_frame
import news_processing

st.set_page_config(layout="wide")
st.header('Contrastive Keyword Extraction')
conn_news = sqlite3.connect('../datasets/ap-matched-sentences.db')
pd.set_option('display.max_columns', None)


def get_doc(article_id=17313):
    data_news = news_processing.create_data(article_id, conn_news)
            
    documents = news_processing.get_documents(data_news)

    return documents

def added_df(added):
    return pd.DataFrame({"sentence": added.keys(), "added": added.values()}).reset_index(drop=True)
#print(documents[0])



def display_keywords(keywords, k):
    inter_kws, inter_scores, delta_int = create_inter_frame(keywords)
    df = pd.DataFrame({'delta': delta_int, 'keyword': inter_kws, 'score': inter_scores})
    
    return df.head(k)

article_id = st.selectbox(
    'Choose an article',
    (17313, 17348, 16832, 17313))




col1, col2 = st.columns(2)



documents = get_doc(article_id)

with col1:
    ngram = st.slider("Max Ngram:", 1, 6)
    former = st.text_area('Orignal Version: ', documents[0], height=400)
    

with col2:
    top_k = st.slider("Top-k Keywords:", 5, 30)
    later = st.text_area('Later Version: ', documents[-1], height=400)
    

run = st.button('run')

if run:
    keywords, matched_dicts, changed_sentences, added, deleted = contrastive_extraction([former, later], 
                                                                        max_ngram=ngram,
                                                                        min_ngram=1, 
                                                                        show_changes=False, 
                                                                        symbols_to_remove=string.punctuation,
                                                                        importance_estimator=sentence_importance.yake_weighted_importance)


    st.write('Keywords:')
    st.table(display_keywords(keywords, top_k))

    st.write('Added Content')
    st.dataframe(added_df(added[0]))