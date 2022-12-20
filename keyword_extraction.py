import pke
import spacy
#nlp = spacy.load("en_core_web_sm")
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import yake

def keybert(documents, ngram_range=(1, 1), stop_words=None, top_n=5, model=None, diversity=0.5, use_mmr=False,
           use_maxsum=True, nr_candidates=20, highlight=False):
    """Extract Keywords from one or more Documents.
    
    Extracts Keywords from n input Documents and puts them in to
    dictonary, where the keys are the version number of the document
    and the values the corresponding keywords for that version.

    Args:
        documents (list) : List of Strings/Documents
        nr_candidates (int): Number of candidates for max sum distance
        diversity (float) : Diversity for MMR the higher the more diverse
        model (string / model) : Any model from sentenceTransomers or hugging face Transformers
        
    Returns:
        Dictonary of Keywords
    """
    number_of_documents = len(documents)
    kws = {version:[] for version in range(number_of_documents)}
    kw_model = KeyBERT(model=model)
    for current in range(number_of_documents):
        kws[current] = kw_model.extract_keywords(documents[current],
                                                 keyphrase_ngram_range=ngram_range,
                                                 stop_words=stop_words,
                                                 top_n=top_n,
                                                 use_mmr=use_mmr,
                                                 diversity=diversity,
                                                 use_maxsum=use_maxsum,
                                                 nr_candidates=nr_candidates,
                                                 highlight=highlight
                                                )
        
    return kws
    

def extract_current(document, extractor=pke.unsupervised.YAKE, lang="en", n_best=10):
    extractor = extractor()
    extractor.load_document(input=document, language=lang)
    
    extractor.candidate_selection()

    extractor.candidate_weighting()

    keyphrases = extractor.get_n_best(n_best)
    
    return keyphrases

def print_current_keywords(keywords):
    for keyword in keywords:
        print(keyword)
    print("\n")    
    
def extract(documents, extractor=pke.unsupervised.YAKE, lang="en", n_best=10):
    number_of_documents = len(documents)
    kws = {version:[] for version in range(number_of_documents)}
    for current in range(number_of_documents):
        kws[current] = extract_current(documents[current], extractor,lang=lang, n_best=n_best)
        
    return kws

def extract_yake(documents, language="en", max_ngram_size=1, 
                 deduplication_threshold = 0.9, deduplication_algo = 'seqm',
                 windowSize=1,numOfKeywords=10):
    number_of_documents = len(documents)
    kws = {version:[] for version in range(number_of_documents)}
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, 
                                                dedupLim=deduplication_threshold,dedupFunc=deduplication_algo, 
                                                windowsSize=windowSize, top=numOfKeywords, features=None)
    
    for current in range(number_of_documents):
        kws[current] = custom_kw_extractor.extract_keywords(documents[current])
            
    return kws
        
def print_keywords(dict_of_keywords):
    for (version, keywords) in dict_of_keywords.items():
        print(f"Version: {version}")
        print_current_keywords(keywords)
    
    
def get_keyword_names(keywords, n):
    return [keyword for (keyword, _) in keywords[n]]

def get_keyword_values(keywords, n):
    return [values for (_, values) in keywords[n]]

def keyword_as_set(keywords, version):
    return set([keyword for (keyword, _) in keywords[version]])

def intersection(keywords, version_x, version_y):
    x = keyword_as_set(keywords, version_x)
    y = keyword_as_set(keywords, version_y)
    return list(x & y)

def intersection_with_importance(keywords, version_x, version_y):
    intersec = intersection(keywords, version_x , version_y)
    kw_x = [(keyword, values) for (keyword, values) in keywords[version_x] if keyword in intersec]
    kw_y = [(keyword, values) for (keyword, values) in keywords[version_y] if keyword in intersec]
    return (kw_x, kw_y)

def diff(keywords, version_x, version_y):
    x = keyword_as_set(keywords, version_x)
    y = keyword_as_set(keywords, version_y)
    return list(x - y)

def diff_keywords(keywords, version_x, version_y):
    return (diff(keywords, version_x, version_y), diff(keywords, version_y, version_x))

def keyword_summary(keywords):
    num_keywords = len(keywords)
    for i in range(num_keywords):
        for j in range(i+1, num_keywords):
            only_i, only_j = diff_keywords(keywords, i, j)
            both_i, both_j = intersection_with_importance(keywords, i, j)
            print(f"Comparision between keywords of version {i} and version {j}")
            print(f"Only in version {i}: {only_i}")
            print("In Both versions:")
            print(f"Version {i}: {both_i}")
            print(f"Version {j}: {both_j}")
            print(f"Only in version {j}: {only_j}")
            print("\n")
            
                      
def create_embeddings(keywords, model="distilbert-base-nli-mean-tokens"):
    model = SentenceTransformer(model)
    number_of_documents = len(keywords)
    keyword_embeddings = {version:[] for version in range(number_of_documents)}
    
    for current in range(number_of_documents):
        keyword_embeddings[current] = model.encode(get_keyword_names(keywords, current))
        
    return keyword_embeddings


def cos_similarity(keywords, embeddings, version_x, version_y):
    cos = cosine_similarity(embeddings[version_x], embeddings[version_y])
    cos_frame = pd.DataFrame(cos, get_keyword_names(keywords, version_x), get_keyword_names(keywords, version_y))
    return cos_frame


def show_cosine_similarities(keywords, embeddings):
    number_of_documents = len(embeddings)
    for i in range(number_of_documents):
        for j in range(i+1, number_of_documents):
            cosinus = cos_similarity(keywords, embeddings, i, j)
            display(cosinus.style
                    .set_table_attributes("style='display:inline'")
                    .set_caption(f"Cosinus Similarity Matrix between Versions {i} and {j}"))
            
            
