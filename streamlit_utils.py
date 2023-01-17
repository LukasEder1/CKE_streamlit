import pandas as pd
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

def annotate_keywords(document, intermediate_keywords, changed_indices, matched_dict, new, ngram, added):
    sentences = nltk.sent_tokenize(document.replace("$", "&#36;"))
    words = [nltk.word_tokenize(sentences[i])  for i in range(len(sentences))]
    matched_and_changed = [matched_dict[i][0][0] for i in changed_indices]
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
        if i in matched_and_changed:
            annotated_text(*words[i])
        elif i in new:
            annotated_text((sentences[i], "new"))
        else:
            st.write(sentences[i])

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