from contrastive_keyword_extraction import contrastive_extraction, final_score, combine_keywords
import sqlite3
import pandas as pd
import string
import sentence_importance
import sentence_comparision
import streamlit as st
import nltk
import numpy as np
import re
import pickle
import time

def list_to_string(l):
    text = ""
    for i in new[0][:-2]:
        text += str(i) + ", "

    text += str(new[0][-1])
    return text

def get_matched_indices(additions, matched_dict):
    indices = []
    scores = []
    for addition in additions.keys():
        mathed_idx, score = matched_dict[addition][0]
        
        indices.append(int(mathed_idx))
        
        scores.append(float(score))
    
    return indices, scores

def create_inter_frame(inter_keywords):
    keywords_dict = list(inter_keywords.values())
    kws = []
    scores = []
    for i in range(len(inter_keywords)):

        for kw, score in keywords_dict[i].items():
            kws.append(kw)
            scores.append(score)
    
    return kws, scores

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
    css_string += f" b.new {{background-color: rgb(0, 255, 204);}}"
    for label, score in keywords.items():
        css_string += f" b.{label} {{background-color: rgb(0,{colouring[label]},0);}}"

    css_string += " </style>"
    
    return css_string


def highlight_keywords(document, intermediate_keywords, changed_indices, matched_dict, new, ngram, added):
    sentences = nltk.sent_tokenize(document.replace("$", "&#36;"))
    
    print(sentences)
    g_values = colour_map(intermediate_keywords, len(intermediate_keywords))
    
    highlighted_string = sentences.copy()

    for i in changed_indices:
        
        matched_idx, _ = matched_dict[i][0]

        sentence = sentences[int(matched_idx)]
        
        for keyword in intermediate_keywords.keys():
            if keyword.lower() in added[i]:
                
                sentence = re.sub(keyword.lower(), 
                    f"<b class=\"{keyword.lower()}\">" +  keyword +"</b>",
                sentence, flags=re.I)

        highlighted_string[matched_idx] = sentence

    for j in new:
        highlighted_string[j] = f"<b class=\"new\">" + sentences[j] + "</b>"

    html_string = " ".join(highlighted_string)
    
    html_string += create_stylesheet(intermediate_keywords, g_values)
    
    return html_string

with open("docs.pkl", "rb") as file:
    # read list from file
    docs = pickle.load(file)

# BEGIN DOCUMENT
st.set_page_config(layout="wide")
st.header('Contrastive Keyword Extraction')

pd.set_option('display.max_columns', None)


def changed_df(added, matched_dict, deleted):
    matched_indices, scores = get_matched_indices(added, matched_dict)
    return pd.DataFrame({"position original document": added.keys(), 
        "matched position": matched_indices,
        "semantic similarity":scores,
        "added": added.values(),
        "deleted": deleted.values() }).reset_index(drop=True)




def display_keywords(keywords, k):
    inter_kws, inter_scores = create_inter_frame(keywords)
    df = pd.DataFrame({'keyword': inter_kws, 'score': inter_scores})
    
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


lower_bound = st.slider("semantic matching threshold", 0.0, 1.0, 0.6)

col1, col2 = st.columns(2)


documents = docs[article_id]

with col1:
    ie = st.selectbox(
    'Importance Estimator',
    ('TextRank', 'Yake Weighted Keyword Count', 'Yake Unweighted Keyword Count'))
    ngram = st.slider("Max Ngram:", 1, 10, 2)
    former = st.text_area('Original Version: ', documents[0], height=400)
    

with col2:
    match = st.selectbox(
    'Matching Algorithm',
    ('Semantic Search', 'Weighted tfidf'))
    top_k = st.slider("Top-k Keywords:", 5, 30, 10)
    later = st.text_area('Later Version: ', documents[-1], height=400)

run = st.button('Compare Documents')

if run:
    keywords, matched_dicts, changed_sentences, added, deleted, new = contrastive_extraction([former, later], 
                                                                        max_ngram=ngram,
                                                                        min_ngram=1, 
                                                                        show_changes=False, 
                                                                        symbols_to_remove=string.punctuation,
                                                                        importance_estimator=ies[ie],
                                                                        match_sentences=matchers[match],
                                                                        threshold=lower_bound)


    st.write('Keywords:')
    st.table(display_keywords(keywords, top_k))

    st.write('Added Content')
    st.dataframe(changed_df(added[0], matched_dicts[0], deleted[0]), use_container_width=True)
    
    

    st.write(f"New sentences in later version: {list_to_string(new[0])}")

    kws = keywords[0]
    
    html_string1 = highlight_keywords(later, 
                                    {k: kws[k] for k in list(kws)[:top_k]},
                                    changed_sentences[0],
                                    matched_dicts[0],
                                    new[0],
                                    added=added[0], 
                                    ngram=1)

    st.write("Monograms Highlighted")
    st.markdown(html_string1, unsafe_allow_html=True)
