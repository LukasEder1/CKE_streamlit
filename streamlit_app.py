from contrastive_keyword_extraction import contrastive_extraction 
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
from annotated_text import annotated_text
from streamlit_utils import *
import difflib
import utilities

with open("docs.pkl", "rb") as file:
    # read list from file
    docs = pickle.load(file)


# BEGIN DOCUMENT
st.set_page_config(page_title="CKE", page_icon=":shark:", layout="wide")
st.header('Contrastive Keyword Extraction')

pd.set_option('display.max_columns', None)


nltk.download("punkt")
nltk.download('stopwords')


article_id = st.selectbox(
    'Choose a Document or try it with your own one.',
    ("Custom", "Example 0", "Example 1", "Article 16159", "Article 17313", 
    "Article 17748","Policy 99880", "Policy 90232", "Policy 98447",
     "Policy 106601", "Policy 106604"))



ies = {"TextRank":sentence_importance.text_rank_importance,
      "Yake Weighted Keyword Count": sentence_importance.yake_weighted_importance,
      "Yake Unweighted Keyword Count": sentence_importance.yake_unweighted_importance
      }


matchers = {"Semantic Search": sentence_comparision.match_sentences_semantic_search,
            "Weighted tfidf": sentence_comparision.match_sentences_tfidf_weighted}

combinator = {"Linear Combination": utilities.alpha_combination,
              "Geometric Combination": utilities.gamma_combination,
              "Harmonic Mean": utilities.harmonic_mean}

stopwords_collection = {"NLTK English Stopwords": nltk.corpus.stopwords.words("english"),
            "None": []}

col1, col2 = st.columns(2)

if article_id == "Custom":
    documents = ["", ""]
else:
    documents = docs[int(article_id.split(" ")[1])]


with col1:

    # Ngram Size of the Keywords
    ngram = st.slider("Max Ngram:", 1, 10, 2)
    
    former = st.text_area('Original Version: ', documents[0], height=400)
    

with col2:
    
    # The best k Keywords
    top_k = st.slider("Top-k Keywords:", 5, 30, 10)

    later = st.text_area('Latter Version: ', documents[-1], height=400)

with st.expander("Advanced Settings"):

    # How to evaluate the Importance of Sentences
    ie = st.selectbox(
    'Importance Estimator',
    ('TextRank', 'Yake Weighted Keyword Count', 'Yake Unweighted Keyword Count'))

    # How to match Sentences
    match = st.selectbox(
    'Matching Algorithm',
    ('Semantic Search', 'Weighted tfidf'))

    comb = st.selectbox(
    'Combinator of Sentence Importance and Change Importance',
    ('Linear Combination', 'Geometric Combination', 'Harmonic Mean'))

    param = 0.5

    if comb == "Linear Combination":
       param = st.slider("Alpha", 0.0, 1.0, 0.5) 
    
    if comb == "Geometric Combination":
        param = st.slider("Gamma", 0.0, 1.0, 0.5)

    # Lower Bound for matching sentences
    lower_bound = st.slider("Semantic matching threshold", 0.0, 1.0, 0.6)
    col_stop1, col_stop2 = st.columns(2)
    with col_stop1:
        use_stopwords = st.selectbox(
    'Preset Stopword Collection',
    ('NLTK English Stopwords', 'None'))
    # Include extra stopwords
    with col_stop2:
        extra_stopwords = st.multiselect("Remove additional stopwords",
                                list(set(nltk.word_tokenize(later)).union(set(nltk.word_tokenize(former)))),
                                [])

    # Display Ngrams
    show_grams = st.checkbox('Show monograms in context (experimental feature)')

    # Display Importance of Sentences
    show_importance = st.checkbox('Show Sentence Ranking (depends on the Importance Estimator)')

    num_splits = st.number_input("Number of Splits allowed", 1, 4, 1)

run = st.button('Compare Documents')

# BEGIN: Display Results
if run:

    # Check that Text Fields are not empty
    if is_empty(former) or is_empty(later):
        st.error('Please make sure that none of the Documents are empty.')

    
    else:
        sw = stopwords_collection[use_stopwords] + [stopword.lower() for stopword in extra_stopwords]

        keywords, matched_dict, changed_sentences, added, deleted, new, ranking, removed,matched_indices, ud = contrastive_extraction([former, later], 
                                                                            max_ngram=ngram,
                                                                            min_ngram=1, 
                                                                            show_changes=False, 
                                                                            symbols_to_remove=string.punctuation,
                                                                            importance_estimator=ies[ie],
                                                                            match_sentences=matchers[match],
                                                                            threshold=lower_bound,
                                                                            extra_stopwords=sw,
                                                                            top_k=int(num_splits),
                                                                            combinator=combinator[comb],
                                                                            alpha_gamma=float(param))

        
        st.markdown("<h1 style='text-align: center;'>Diff-Content and Matched Sentences</h1>", unsafe_allow_html=True)
        st.dataframe(changed_df(added, matched_dict, deleted), use_container_width=True)
        st.markdown("<h1 style='text-align: center;'>Contrastive Keywords</h1>", unsafe_allow_html=True)
        st.table(display_keywords(keywords, top_k))
        kws = keywords

        # Highlight Contrastive Keywords in Context        
        if show_grams:
            st.markdown("<h1 style='text-align: center;'>Keywords in Context</h1>", unsafe_allow_html=True)
            annotate_keywords(later, 
                            {k: kws[k] for k in list(kws)[:top_k]},
                            changed_sentences,
                            matched_dict,
                            new,
                            added=added, 
                            ngram=1)

        # Highlight matched/deleted/added and split sentences
        highlight_changes(former,
                                later,
                                changed_sentences,
                                matched_dict,
                                new,
                                removed,
                                matched_indices)

        # Show and Compare the most Important Sentences
        if show_importance:
            show_sentence_importances(ranking, former, later)
