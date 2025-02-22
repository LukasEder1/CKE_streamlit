import pandas as pd
import streamlit as st
import re
import pysbd
from TextHighlight import ContrastiveTextHighlighter

def drop_dupilcates(list):
    result = []
    for i in list:
        if i not in result:
            result.append(i)
    
    return result

def get_ngrams_of_size_n(diff_content, ngram_size):
    return [ngram for ngram in diff_content if len(ngram.split(" ")) == ngram_size]


def get_matched_indices(matched_dict):
    """ Get indices of matched Sentences

    Args:
        matched_dict (dict):Keys: Indices of Document A, 
                            Values: List of Pairs <Index of Document B| semantic similarity>

    Returns:
        List of all sentences in version B, that have been matched to
    """
    return [i for i in list(matched_dict.keys()) if len(matched_dict[i]) > 0]

def is_empty(document):
    return len(document) == 0 or document.isspace()

def create_keyword_frame(keywords):
    """ Cast kw-dictonary to pandas DataFrame

    Args:
        keywords (dict): Keyword, Score Pairs

    Returns:
        DataFrame with Keyword, Score Columns
    
    """
    kws = []
    scores = []

    for kw, score in keywords.items():
        kws.append(kw)
        scores.append(score)
    
    return kws, scores

def changed_df(added, matched_dict, deleted):
    original_indices = []
    matched_indices = []
    matched_score = []
    added_list = []
    deleted_list = []

    for i in get_matched_indices(matched_dict):
        original_indices += len(matched_dict[i]) * [i]
        for idx, score in  matched_dict[i]:
            matched_indices.append(int(idx))
            matched_score.append(float(score))
            
            added_list.append(added[i].get(int(idx), []))
            deleted_list.append(deleted[i].get(int(idx), []))

    #matched_indices, scores = get_matched_indices(added, matched_dict)
    
    return pd.DataFrame({"original sentence position": original_indices, 
        "matched sentence position": matched_indices,
        "semantic similarity":matched_score,
        "added": added_list,
        "deleted": deleted_list }).reset_index(drop=True)


def create_ranking_df(rank):

    return pd.DataFrame({"Sentence Position": list(rank.keys()),
                        "Importance Score": list(rank.values())})


def display_keywords(keywords, former, latter, k):
    kws, scores = create_keyword_frame(keywords)

    former_kws, former_scores = create_keyword_frame(former)
    latter_kws, latter_scores = create_keyword_frame(latter)

    df_former = pd.DataFrame({'keyword': former_kws, 'score': former_scores})
    df_latter = pd.DataFrame({'keyword': latter_kws, 'score': latter_scores})
    df = pd.DataFrame({'keyword': kws, 'score': scores})
    
    col_former, col_latter = st.columns(2)

    with col_former:
        st.markdown("<h3 style='text-align: center;'>Keywords of Older Version</h3>", unsafe_allow_html=True)
        st.table(df_former.head(k))

    with col_latter:
        st.markdown("<h3 style='text-align: center;'>Keywords of Newer Version</h3>", unsafe_allow_html=True)
        st.table(df_latter.head(k))

    st.markdown("<h3 style='text-align: center;'>Combined Keywords</h3>", unsafe_allow_html=True)
    st.table(df.head(k))

def find_merges(matched_dict):
    merges = {}
    maximum_merge = {}
    maximum_score = {}
    nonmaximum_mapping = {}

    for k in matched_dict.keys():
        for idx, score in matched_dict[k]:

            merges[int(idx)] = merges.get(int(idx), []) + [k]
            if float(score) > maximum_score.get(int(idx), 0.0):

                maximum_score[int(idx)] = float(score)
                maximum_merge[int(idx)] = k

    for i in merges.keys():
        for j in merges[i]:
            if j != maximum_merge[i]:

                nonmaximum_mapping[j] = i

    return merges, maximum_merge, nonmaximum_mapping

def find_max_indices(matched_dict, matched_indices, changed_indices):
    max_index = {i:int(matched_dict[i][0][0]) for i in changed_indices}
    max_score = {i:int(matched_dict[i][0][1]) for i in changed_indices}
    nonmax_mapping = {}

    for i in set(changed_indices):
        if len(matched_dict[i]) > 1:
            for j, score in matched_dict[i]:
                j = int(j)
                
                if score > max_score[i]:
                    
                    max_index[i] = int(j)
                    max_score[i] = score    

        
        for idx, _ in matched_dict[i]:
            if int(idx) != max_index[i]:
                nonmax_mapping[int(idx)] = max_index[i]


    return max_index, nonmax_mapping

def show_sentence_importances(ranking, former, later):

    # Sentence Boundary Detecton (Tokenization)
    seg = pysbd.Segmenter(language="en", clean=False)
    former_sentences = seg.segment(former)

    later_sentences = seg.segment(later)

    
    ranking_earlier = create_ranking_df(ranking[0])
    ranking_latter = create_ranking_df(ranking[1])

    former_html = []
    latter_html = []
    
    for i in list(ranking[0].keys()):
        former_html.append(annotate_sentence(former_sentences[i], type="changed", label=f"Index: {i}  Importance: {round(ranking[0][i], 4)}"))

 
    for i in list(ranking[1].keys()):
        latter_html.append(annotate_sentence(later_sentences[i], type="changed", label=f"Index: {i}  Importance: {round(ranking[1][i], 4)}"))

    return ranking_earlier, ranking_latter, former_html, latter_html

def union_of_words(former, latter):
    former = re.findall(r"[\w']+", former.lower())
    latter = re.findall(r"[\w']+", latter.lower())
    return sorted(list(set(latter).union(set(former))))


def add_lines(n):
    for _ in range(n):
        st.write("")

def annotate_sentence(sentence, type, label):
    return f'''<p><span class={type}>{sentence}<span class="index">{label}</span></span></p>'''



def highlight_custom_changes(former, later, changed_indices, matched_dict, new, removed, matched_indices, n_gram, kw_f, kw_l, top_k, highlight_kws=True):

    # Setup <- Change Colour

    # Find the index with the highest Semantic Similarity for all split sentences
    max_index, nonmax_mapping = find_max_indices(matched_dict, matched_indices, changed_indices)
    merges, maximum_merges, merges_mapping = find_merges(matched_dict)
    seg = pysbd.Segmenter(language="en", clean=False)
    
    if highlight_kws:
        th_former = ContrastiveTextHighlighter(max_ngram_size = n_gram, rgb= (204, 0, 0), top_k=top_k)
        th_latter = ContrastiveTextHighlighter(max_ngram_size = n_gram, rgb= (0, 102, 0), top_k=top_k)
    former_sentences = seg.segment(former)
    if highlight_kws:
        if len(kw_f) > 0:
            former_sentences = [th_former.highlight(sentence, kw_f) for sentence in former_sentences]

    later_sentences = seg.segment(later)
    if highlight_kws:
        if len(kw_l) > 0:
            later_sentences = [th_latter.highlight(sentence, kw_l) for sentence in later_sentences]

    # find sentences in newer version that have been matched to
    # and where the syntatic similarity is below 1.0
    matched_and_changed = [matched_dict[i][0][0] for i in changed_indices]
    
    # includes the split to sentences, without the max split sentence
    splits = []
    max_splits = []

    # find non-max split sentences
    for i, l in matched_dict.items():
        
        if len(l) > 1:
            max_splits.append(int(l[0][0]))

    for l in matched_dict.values():
        for i, score in l[1:]:
            if i not in max_splits and score < 1:
                splits.append(int(i))


    nonmax_merge = list(merges_mapping.keys())

    # annotate
    later_html = []
    
    d = {int(i):count for count, i in enumerate(drop_dupilcates(matched_and_changed))}
    
    for i in range(len(later_sentences)):
        if i in new:
            later_html.append(annotate_sentence(later_sentences[i], "new", "New"))
        elif i in matched_and_changed:
            later_html.append(annotate_sentence(later_sentences[i], type="changed", label=f"{d[i]}"))
        elif i in splits:
            later_html.append(annotate_sentence(later_sentences[i], type="changed", label=f"{d[nonmax_mapping[i]]} - split"))
        else:
            later_html.append(later_sentences[i])


    former_html = []
    # use the annotated component to highlight text
    for i in range(len(former_sentences)):
        if i in removed:
            former_html.append(annotate_sentence(former_sentences[i], "removed", "Removed"))
        elif i in changed_indices:
            """
            if not i in nonmax_merge:
                annotated_text((former_sentences[i], f"{matched_dict[i][0][0]}", "#f2f2f2"))
            else:
                annotated_text((former_sentences[i], f"{matched_dict[i][0][0]} - merge", "#f2f2f2"))
            """
            former_html.append(annotate_sentence(former_sentences[i], type="changed", label=f"{d[int(matched_dict[i][0][0])]}"))


            
        else:
            former_html.append("<p>"+former_sentences[i]+"</p>")

    return former_html, later_html


