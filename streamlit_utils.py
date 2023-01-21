import pandas as pd
import streamlit as st
import nltk
import numpy as np
import re
import pickle
from annotated_text import annotated_text

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

def find_max_indices(matched_dict, matched_indices, changed_indices):
    max_index = {i:int(matched_dict[i][0][0]) for i in changed_indices}
    max_score = {i:int(matched_dict[i][0][1]) for i in changed_indices}
    #splits = []
    for i in changed_indices:
        if len(matched_dict[i]) > 1:
            for j, score in matched_dict[i]:
                j = int(j)
                
                if score > max_score[i]:
                    
                    max_index[i] = int(j)
                    max_score[i] = score    
    
    return max_index

def highlight_changes(former, later, changed_indices, matched_dict, new, removed, matched_indices):

    # Setup
    max_index = find_max_indices(matched_dict, matched_indices, changed_indices)

    former_sentences = nltk.sent_tokenize(former.replace("$", "&#36;"))

    later_sentences = nltk.sent_tokenize(later.replace("$", "&#36;"))

    matched_and_changed = [matched_dict[i][0][0] for i in changed_indices]
    
    splits = []

    for i, l in matched_dict.items():
        
        if len(l) > 1:
            for j, score in l:
                if int(j) not in list(max_index.values()):
                    splits.append(int(j))
                    #split_from.update({str(j): max_index[i]})

    # Begin Document

    st.markdown("<h1 style='text-align: center;'>Sentence Matching Results</h1>", unsafe_allow_html=True)

    col_former, col_later = st.columns(2)
    with col_former:
        st.markdown("<h2 style='text-align: center;'>Original Document</h2>", unsafe_allow_html=True)

        # use the annotated component to highlight text
        for i in range(len(former_sentences)):
            if i in removed:
                annotated_text((former_sentences[i], "deleted", "#ff6666"))
            elif i in changed_indices:
                annotated_text((former_sentences[i], f"{matched_dict[i][0][0]}", "#f2f2f2"))
            else:
                st.write(former_sentences[i])

    with col_later:
        st.markdown("<h2 style='text-align: center;'>Latter Document</h2>", unsafe_allow_html=True)

        for i in range(len(later_sentences)):
            if i in new:
                annotated_text((later_sentences[i], "new", "#4dff4d"))
            elif i in splits:
                annotated_text((later_sentences[i], f"{str(i)} - split", "#f2f2f2"))
            elif i in matched_and_changed:
                annotated_text((later_sentences[i], f"{i}", "#f2f2f2"))
            else:
                st.write(later_sentences[i])


def show_sentence_importances(ranking, former, later):

    st.markdown("<h1 style='text-align: center;'>Sentence Importance Calculation</h1>", unsafe_allow_html=True)

    former_sentences = nltk.sent_tokenize(former.replace("$", "&#36;"))

    later_sentences = nltk.sent_tokenize(later.replace("$", "&#36;"))

    rcol1, rcol2 = st.columns(2)
    ranking_earlier = create_ranking_df(ranking[0])
    ranking_latter = create_ranking_df(ranking[1])
    with rcol1:
        st.markdown("<h3 style='text-align: center;'>Original Document</h3>", unsafe_allow_html=True)
        st.table(ranking_earlier)
        for i in list(ranking[0].keys()):
            annotated_text((former_sentences[i], f"{round(ranking[0][i], 4)}", "#f2f2f2"))

    with rcol2:
        st.markdown("<h3 style='text-align: center;'>Latter Document</h3>", unsafe_allow_html=True)
        st.table(ranking_latter)
        for i in list(ranking[1].keys()):
            annotated_text((later_sentences[i], f"{round(ranking[1][i], 4)}", "#f2f2f2"))