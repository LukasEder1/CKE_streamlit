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
from annotated_text import annotated_text
from streamlit_utils import *

with open("docs.pkl", "rb") as file:
    # read list from file
    docs = pickle.load(file)

# custom examples <- refactor
docs[0] = ["In this paper, we introduce TextRank – a graph-based ranking model for text processing, and show how this model can be successfully used in natural language applications. In particular, we propose two innovative unsupervised methods for keyword and sentence extraction, and show that the results obtained compare favorably with previously published results on established benchmark.", "TextRank, a graph-based ranking system, is introduced in this paper. Ranking model for text processing, and demonstrate how this model can be used successfully in natural language processing applications. We propose two novel unsupervised methods for keyword and sentence extraction in particular, and demonstrate that the results obtained compare favorably with previously published results on established benchmarks."]
docs[1] = ["""The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data.""",
 """The dominant sequence transduction fashions are primarily based on complicated recurrent or
convolutional neural networks that consist of an encoder and a decoder. The best
performing fashions additionally join the encoder and decoder via an attention
mechanism. We endorse a new easy community architecture, the Transformer,
based entirely on interest mechanisms, shelling out with recurrence and convolutions
entirely. Experiments on two computer translation duties exhibit these fashions to
be optimum in nice whilst being greater parallelizable and requiring significantly
less time to train. Our mannequin achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, enhancing over the present first-rate results, including
ensembles, by using over two BLEU. On the WMT 2014 English-to-French translation task,
our mannequin establishes a new single-model modern day BLEU rating of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the education expenses of the
best fashions from the literature. We show that the Transformer generalizes nicely to
other duties by means of making use of it effectively to English constituency parsing each with
large and confined education data. 
"""]

# BEGIN DOCUMENT
st.set_page_config(page_title="CKE", page_icon=":shark:", layout="wide")
st.header('Contrastive Keyword Extraction')

pd.set_option('display.max_columns', None)


nltk.download("punkt")
nltk.download('stopwords')

article_id = st.selectbox(
    'Choose a Document or try it with your own one.',
    ("Custom", "Example 0", "Example 1", "Article 16159", "Article 17313", 
    "Article 17748", "Policy 90232", "Policy 98447",
     "Policy 106601", "Policy 106604"))



ies = {"TextRank":sentence_importance.text_rank_importance,
      "Yake Weighted Keyword Count": sentence_importance.yake_weighted_importance,
      "Yake Unweighted Keyword Count": sentence_importance.yake_unweighted_importance
      }

matchers = {"Semantic Search": sentence_comparision.match_sentences_semantic_search,
            "Weighted tfidf": sentence_comparision.match_sentences_tfidf_weighted}



col1, col2 = st.columns(2)

if article_id == "Custom":
    documents = ["", ""]
else:
    documents = docs[int(article_id.split(" ")[1])]


with col1:

    ngram = st.slider("Max Ngram:", 1, 10, 2)
    former = st.text_area('Original Version: ', documents[0], height=400)
    

with col2:
    
    top_k = st.slider("Top-k Keywords:", 5, 30, 10)
    later = st.text_area('Latter Version: ', documents[-1], height=400)

with st.expander("Advanced Settings"):
    ie = st.selectbox(
    'Importance Estimator',
    ('TextRank', 'Yake Weighted Keyword Count', 'Yake Unweighted Keyword Count'))

    match = st.selectbox(
    'Matching Algorithm',
    ('Semantic Search', 'Weighted tfidf'))

    lower_bound = st.slider("Semantic matching threshold", 0.0, 1.0, 0.6)

    stopwords = st.multiselect("Remove additional stopwords",
                            list(set(nltk.word_tokenize(later))),
                            [])

    show_grams = st.checkbox('Show monograms in context (experimental feature)')
    show_importance = st.checkbox('Show Sentence Ranking (depends on the Importance Estimator)')

run = st.button('Compare Documents')

if run:
    if is_empty(former) or is_empty(later):
        st.error('Please make sure that none of the Documents are empty.')

    
    else:
        keywords, matched_dicts, changed_sentences, added, deleted, new, ranking, removed = contrastive_extraction([former, later], 
                                                                            max_ngram=ngram,
                                                                            min_ngram=1, 
                                                                            show_changes=False, 
                                                                            symbols_to_remove=string.punctuation,
                                                                            importance_estimator=ies[ie],
                                                                            match_sentences=matchers[match],
                                                                            threshold=lower_bound,
                                                                            extra_stopwords=[stopword.lower() for stopword in stopwords])

        st.markdown("<h1 style='text-align: center;'>Diff-Content and Matched Sentences</h1>", unsafe_allow_html=True)
        st.dataframe(changed_df(added[0], matched_dicts[0], deleted[0]))
        
        st.markdown("<h1 style='text-align: center;'>Contrastive Keywords</h1>", unsafe_allow_html=True)
        st.table(display_keywords(keywords, top_k))

        
        #st.write(f"New sentences in the later version are marked as light blue in the following text. The indices of the new sentence are: {list_to_string(new[0])}.")

        kws = keywords[0]
        
        if show_grams:
            st.markdown("<h1 style='text-align: center;'>Keywords in Context</h1>", unsafe_allow_html=True)
            annotate_keywords(later, 
                            {k: kws[k] for k in list(kws)[:top_k]},
                            changed_sentences[0],
                            matched_dicts[0],
                            new[0],
                            added=added[0], 
                            ngram=1)


        #htm = highlight_earlier(former, changed_sentences[0], list(deleted[0].keys()))
        
        #st.markdown(htm, unsafe_allow_html=True)
        if show_importance:
            show_sentence_importances(ranking, former, later)

        highlight_changes(former,
                                later,
                                changed_sentences[0],
                                matched_dicts[0],
                                new[0],
                                removed[0])