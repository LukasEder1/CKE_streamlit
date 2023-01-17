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

def is_empty(document):
    return len(document) == 0 or document.isspace()

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
    
    highlighted_string = []

    for i in changed_indices:
        
        matched_idx, _ = matched_dict[i][0]

        sentence = sentences[int(matched_idx)]
        

        for keyword in intermediate_keywords.keys():
            if keyword.lower() in added[i]:
                
                sentence = re.sub(keyword.lower(), 
                    f"(\"" +  keyword + "\" , \"verb\")",
                sentence, flags=re.I)
        annotated_text(*sentence)
    """
    for j in new:
        highlighted_string[j] = f"(" + sentences[j] + ", \"\")"

    """

    #html_string = " ".join(highlighted_string)
    
    return ""



def annotate_keywords(document, intermediate_keywords, changed_indices, matched_dict, new, ngram, added):
    sentences = nltk.sent_tokenize(document.replace("$", "&#36;"))
    words = [nltk.word_tokenize(sentences[i])  for i in range(len(sentences))]

    for i in changed_indices:

        matched_idx, _ = matched_dict[i][0]

        sentence = sentences[int(matched_idx)]
        keywords = list(intermediate_keywords.keys())

        for j in range(len(words[matched_idx])):
            word = words[matched_idx][j]

            if type(word) != str:
                break

            if word.lower() in added[i] and word.lower() in keywords:
                words[matched_idx][j] = (f"{word}", f"{round(intermediate_keywords[word.lower()], 5)}")

            else:
                words[matched_idx][j] += " "

    for i in range(len(words)):
        if i in changed_indices:
            annotated_text(*words[i])

        elif i in new:
            annotated_text((sentences[i], "new"))
        else:
            st.write(sentences[i])

def sentence_level_css():
    css_string = "<style>"
    css_string += f" b.del {{background-color: rgb(255, 0, 0);}}"
    css_string += f"b.changed {{background-color: rgb(255, 255, 0);}}"
    css_string += "</style"

    return css_string

def highlight_earlier(document, changed_indices, deleted):

    sentences = nltk.sent_tokenize(document.replace("$", "&#36;"))

    highlighted_string = sentences.copy()

    for i in changed_indices:

        highlighted_string[i] = f"<b class=\"changed\">" + sentences[i] + "</b>"



    html_string = " ".join(highlighted_string)

    html_string += sentence_level_css()

    return html_string

def highlight_latter(document, changed_indices, matched_dict, new):

    sentences = nltk.sent_tokenize(document.replace("$", "&#36;"))

    highlighted_string = sentences.copy()

    for i in changed_indices:
        
        matched_idx, _ = matched_dict[i][0]

        highlighted_string[matched_idx] = f"<b class=\"matched\">" + sentences[int(matched_idx)] + "</b>"



    html_string = " ".join(highlighted_string)

    html_string += sentence_level_css()

    return html_string


def changed_df(added, matched_dict, deleted):
    matched_indices, scores = get_matched_indices(added, matched_dict)
    return pd.DataFrame({"original sentence position": added.keys(), 
        "matched sentence position": matched_indices,
        "semantic similarity":scores,
        "added": added.values(),
        "deleted": deleted.values() }).reset_index(drop=True)


def create_ranking_df(rank):

    return pd.DataFrame({"Sentence Position": list(rank.keys()),
                        "Importance Score": list(rank.values())})

def display_keywords(keywords, k):
    inter_kws, inter_scores = create_inter_frame(keywords)
    df = pd.DataFrame({'keyword': inter_kws, 'score': inter_scores})
    
    return df.head(k)

with open("docs.pkl", "rb") as file:
    # read list from file
    docs = pickle.load(file)

# BEGIN DOCUMENT
st.set_page_config(page_title="CKE", page_icon=":shark:", layout="wide")
st.header('Contrastive Keyword Extraction')

pd.set_option('display.max_columns', None)



article_id = st.selectbox(
    'Choose a Document or try it with your own one.',
    ("Custom", "Article 17313", "Article 16159", "Article 17736", 
    "Article 17748", "Policy 90232", "Policy 98445", "Policy 98447",
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
    later = st.text_area('Later Version: ', documents[-1], height=400)

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
    show_verbose = st.checkbox('Show Sentence Ranking (depends on Importance Estimator)')

run = st.button('Compare Documents')

if run:
    if is_empty(former) or is_empty(later):
        st.error('Please make sure that none of the Documents are empty.')

    
    else:
        keywords, matched_dicts, changed_sentences, added, deleted, new, ranking = contrastive_extraction([former, later], 
                                                                            max_ngram=ngram,
                                                                            min_ngram=1, 
                                                                            show_changes=False, 
                                                                            symbols_to_remove=string.punctuation,
                                                                            importance_estimator=ies[ie],
                                                                            match_sentences=matchers[match],
                                                                            threshold=lower_bound,
                                                                            extra_stopwords=[stopword.lower() for stopword in stopwords])

        st.markdown('# Diff-Content and Matched Sentences')
        st.dataframe(changed_df(added[0], matched_dicts[0], deleted[0]), use_container_width=True)
        
        st.markdown('# Contrastive Keywords')
        st.table(display_keywords(keywords, top_k))

        
        #st.write(f"New sentences in the later version are marked as light blue in the following text. The indices of the new sentence are: {list_to_string(new[0])}.")

        kws = keywords[0]
        
        if show_grams:
            annotate_keywords(later, 
                            {k: kws[k] for k in list(kws)[:top_k]},
                            changed_sentences[0],
                            matched_dicts[0],
                            new[0],
                            added=added[0], 
                            ngram=1)

        #htm = highlight_earlier(former, changed_sentences[0], list(deleted[0].keys()))

        #st.markdown(htm, unsafe_allow_html=True)
        if show_verbose:
            st.markdown("## Sentence Importance Calculation")

            rcol1, rcol2 = st.columns(2)
            ranking_earlier = create_ranking_df(ranking[0])
            ranking_latter = create_ranking_df(ranking[1])
            
            with rcol1:
                st.markdown("Sentence Importance Earlier Version")
                st.table(ranking_earlier)

            with rcol2:
                st.markdown("Sentence Importance Latter Version")
                st.table(ranking_latter)