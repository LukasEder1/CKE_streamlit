from contrastive_keyword_extraction import contrastive_extraction, final_score, combine_keywords
import sqlite3
import pandas as pd
import string
import sentence_importance
import sentence_comparision
import streamlit as st
from baselines import create_inter_frame
import nltk
import numpy as np
import re
import pickle

nltk.download("popular")

def colour_map(keywords, n):
    """
    score -> between 0 -> 1
    """
    colouring = {}
    colours = np.linspace(100, 255, n, endpoint=True)
    
    i = 0
    for kw, score in keywords.items():
        colouring[kw] = colours[i]
        i += 1
    return colouring

def create_stylesheet(keywords, colouring):
    css_string = "<style>"
    for label, score in keywords.items():
        css_string += f" b.{label} {{background-color: rgb(0,{colouring[label]},0);}}"

    css_string += " </style>"
    
    return css_string


def highlight_keywords(document, intermediate_keywords, changed_indices, matched_dict, ngram, added):
    sentences = nltk.sent_tokenize(document)
    
    g_values = colour_map(intermediate_keywords, len(intermediate_keywords))
    
    highlighted_string = ""

    for i in changed_indices:
        
        matched_idx, _ = matched_dict[i][0]

        sentence = sentences[int(matched_idx)]
        
        print("BEFORE \n", sentence)
        for keyword in intermediate_keywords.keys():
            if keyword.lower() in added[i]:
                
                sentence = re.sub(keyword.lower(), 
                    f"<b class=\"{keyword.lower()}\">" +  keyword +"</b>",
                sentence, flags=re.I)

        print("AFTER \n")
        
        highlighted_string += sentence
       
    
    
    print(highlighted_string)
    highlighted_string += create_stylesheet(intermediate_keywords, g_values)
    
    return highlighted_string

with open("docs.pkl", "rb") as file:
    # read list from file
    docs = pickle.load(file)
    
st.set_page_config(layout="wide")
st.header('Contrastive Keyword Extraction')

#conn_news = sqlite3.connect('../datasets/ap-matched-sentences.db')
pd.set_option('display.max_columns', None)


def added_df(added):
    return pd.DataFrame({"sentence": added.keys(), "added": added.values()}).reset_index(drop=True)
#print(documents[0])



def display_keywords(keywords, k):
    inter_kws, inter_scores, delta_int = create_inter_frame(keywords)
    df = pd.DataFrame({'delta': delta_int, 'keyword': inter_kws, 'score': inter_scores})
    
    return df.head(k)

article_id = st.selectbox(
    'Choose an article',
    (17313, 16159, 17736, 17748, 3299, 90232, 98445, 98447, 106601, 106604))


ies = {"TextRank":sentence_importance.text_rank_importance,
      "Yake Weighted Keyword Count": sentence_importance.yake_weighted_importance,
      "Yake Unweighted Keyword Count": sentence_importance.yake_unweighted_importance
      }

matchers = {"Semantic Search": sentence_comparision.match_sentences_semantic_search,
            "Weighted tfidf": sentence_comparision.match_sentences_tfidf_weighted}


col1, col2 = st.columns(2)


documents = doc[article_id]

with col1:
    ie = st.selectbox(
    'Importance Estimator',
    ('TextRank', 'Yake Weighted Keyword Count', 'Yake Unweighted Keyword Count'))
    ngram = st.slider("Max Ngram:", 1, 6)
    former = st.text_area('Orignal Version: ', '', height=400)
    

with col2:
    match = st.selectbox(
    'Matching Algorithm',
    ('Semantic Search', 'Weighted tfidf'))
    top_k = st.slider("Top-k Keywords:", 5, 30)
    later = st.text_area('Later Version: ', '', height=400)
    

run = st.button('Extract Keywords')

if run:
    keywords, matched_dicts, changed_sentences, added, deleted = contrastive_extraction([former, later], 
                                                                        max_ngram=ngram,
                                                                        min_ngram=1, 
                                                                        show_changes=False, 
                                                                        symbols_to_remove=string.punctuation,
                                                                        importance_estimator=ies[ie],
                                                                        match_sentences=matchers[match])


    st.write('Keywords:')
    st.table(display_keywords(keywords, top_k))

    st.write('Added Content')
    st.dataframe(added_df(added[0]), use_container_width=True)
    
    kws = keywords[0]
    
    html_string1 = highlight_keywords(later, 
                                    {k: kws[k] for k in list(kws)[:top_k]},
                                    changed_sentences[0],
                                    matched_dicts[0],
                                    added=added[0], 
                                    ngram=1)

    st.write("Monograms Highlighted")
    st.markdown(html_string1, unsafe_allow_html=True)
