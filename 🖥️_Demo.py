from contrastive_keyword_extraction import contrastive_extraction 
import pandas as pd
import string
import sentence_importance
import sentence_comparision
import streamlit as st
import nltk
import pickle
from streamlit_utils import *
import utilities
from TextHighlight import ContrastiveTextHighlighter

st.set_page_config(page_title="CKE", page_icon=":shark:", layout="wide")

def concat(list):
    return "\n".join(list)

#         border-radius: 5px; /* this gives the rounded look */
#        font-family: arial;
#        font-size: 12px;

def display(html):
    result = f'''<style>

    span.changed {{
        background-color: #f2f2f2;
        color: #000;
        padding: 5px;
        border-radius: 5px;

    }}

    span.new {{
        background-color: #00C300;
        color: #000;
        padding: 5px;
        border-radius: 5px; 
    }}

    span.removed {{
        background-color: #cc0000;
        color: #000;
        padding: 5px;
        border-radius: 5px; 

    }}

    span.kw {{
        padding: 1px;
        padding-left:4px;
        border-radius: 5px; 
        opacity: 1;

    }}

    span.index {{
        //border-left: 1px solid black;
        margin-left: 2px;
        padding-left: 5px;
        padding-right:3px;
        opacity: 0.5;
        text-align: right;
        font-size: 0.9em;
    }}

    </style>
 
    {concat([""] + html)}

    '''

    st.markdown(result, unsafe_allow_html=True)
    
# BEGIN META

st.header('Contrastive Keyword Extraction')

pd.set_option('display.max_columns', None)

st.sidebar.markdown("# Advanced Settings")



nltk.download("punkt")
nltk.download('stopwords')


# Dictonary of preset versioned Documents
with open("docs.pkl", "rb") as file:
    docs = pickle.load(file)    
article_id = st.selectbox(
    'Choose a Document or try it with your own one.',
    ("Custom", "Example 0", "Example 1", "Article 17313", 
    "Article 17748","Policy 99880", "Policy 90232", "Policy 106604",
    "Policy 106601", "Policy 98447"), help="Examples taken from: https://privacypolicies.cs.princeton.edu/ and https://github.com/isi-nlp/NewsEdits")


ies = {"TextRank":sentence_importance.text_rank_importance,
    "Yake Weighted Keyword Count": sentence_importance.yake_weighted_importance
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

# BEGIN GUI
with col1:

    # Ngram Size of the Keywords
    ngram = st.slider("Max Ngram:", 1, 10, 2)
    
    former = st.text_area('Original Version: ', documents[0], height=400)
    

with col2:
    
    # The best k Keywords
    top_k = st.slider("Top-k Keywords:", 5, 30, 10)

    later = st.text_area('Latter Version: ', documents[-1], height=400)


with st.sidebar:

    # How to evaluate the Importance of Sentences
    ie = st.selectbox(
    'Importance Estimator',
    ('TextRank', 'Yake Weighted Keyword Count'))


    match = st.selectbox(
    'Matching Algorithm',
    ('Semantic Search', 'Weighted tfidf'))
    

    # Lower Bound for matching sentences
    lower_bound = st.slider("Semantic Matching Threshold", 0.0, 1.0, 0.6, help='''Acts as a lower bound 
                            for whether or not two sentences match''')


        # How to combine Sentence Importance and Change Importance
    # defined as phi in paper
    comb = st.selectbox(
    'Combinator of Sentence Importance and Change Importance',
    ('Linear Combination', 'Geometric Combination', 'Harmonic Mean'),
    help='''Determines how the importance of a sentence in the original document gets combined with
        the importance of the change between this sentence and its matched counterpart. 
        ''')


    # Corresponding Parameter
    param = 0.5

    if comb == "Linear Combination":
        param = st.slider("Beta", 0.0, 1.0, 0.5) 
    
    if comb == "Geometric Combination":
        param = st.slider("Gamma", 0.0, 1.0, 0.5)


    use_stopwords = st.selectbox(
            'Preset Stopword Collection',
            ('NLTK English Stopwords', 'None'))

    extra_stopwords = st.multiselect("Remove additional stopwords",
                        union_of_words(former, later),
                        [])

       # Display Ngrams
    show_grams = st.checkbox('Show Keywords in context (experimental feature)')

    # Display  Sentence Importances
    show_importance = st.checkbox('Show Sentence Ranking (depends on the Importance Estimator)')

    # Upper bound on the possible number of splits
    num_splits = st.number_input("Number of Splits allowed", 1, 4, 1)

    st.write("")





run = st.button('Compare Documents')


# BEGIN: Display Results
if run:

    # Check that Text Fields are not empty
    if is_empty(former) or is_empty(later):
        st.error('Please make sure that none of the Documents are empty.')

    
    else:
        # combine stopwords from collection and extra stopwords
        sw = stopwords_collection[use_stopwords] + [stopword.lower() for stopword in extra_stopwords]


        # Extract Keywords, and Matched sentences
        keywords, former_keywords, latter_keywords, matched_dict, changed_sentences, added, deleted, new, ranking, removed,matched_indices, ud = contrastive_extraction([former, later], 
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
        display_keywords(keywords, former_keywords, latter_keywords, top_k)

        kws = keywords

        # Highlight Contrastive Keywords in Context        
        


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

        former_html, later_html = highlight_custom_changes(former,
                                later,
                                changed_sentences,
                                matched_dict,
                                new,
                                removed,
                                matched_indices,
                                ngram,
                                former_keywords,
                                latter_keywords,
                                top_k)
        
        if show_grams:
            st.markdown("<h1 style='text-align: center;'>Keywords In Context</h1>", unsafe_allow_html=True)
        
            col_former, col_later = st.columns(2)

            with col_former:
                st.markdown("<h2 style='text-align: center;'>Former Document</h2>", unsafe_allow_html=True)
                display(former_html)

            with col_later:
                st.markdown("<h2 style='text-align: center;'>Latter Document</h2>", unsafe_allow_html=True)
                display(later_html)