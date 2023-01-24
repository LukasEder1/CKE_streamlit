from difflib import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
import utilities
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def opcodes(a, b):
    s = SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print('{:7}   query[{}:{}] --> matched[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))
        
        
def ndiff(a,b):
    print("".join(ndiff(a.splitlines(keepends=True), b.splitlines(keepends=True))), end="")
    
def syntactic_ratio(a, b):
    # match sentence a and b
    s = SequenceMatcher(lambda x: x == " ", a, b)
    
    # return their ratio
    # rule-of-thumb: ratio > 0.6 -> similar
    return s.ratio()

def freqs(list):
    words = {}
    for word in list:
        words[word] = words.get(word, 0) + 1
    return words

"""
# https://stackoverflow.com/questions/10074638/get-added-and-removed-words-between-two-strings
def find_additions_deletions(a, b):
    af = freqs(a.lower().split())
    bf = freqs(b.lower().split())

    removed = []
    added = []

    for key in af:
        num = bf.get(key)
        if num == None:
            if af[key] > 1:
                words = [key]*af[key]
                removed.extend(words)
            else:
                removed.append(key)

    for key in bf:
        num = af.get(key)
        if num == None:
            added.append(key)
        elif num > 1:
            words = [key]*(num-1)
            removed.extend(words)

    return added, removed
"""

""""""
def find_additions_deletions(a, b):

    # init differ
    d = Differ()
    
    # compare the two 
    a = re.findall(r"[\w']+", a.lower())
    b = re.findall(r"[\w']+", b.lower())

    diff = d.compare(a, b)
    changes = [change for change in diff if change.startswith('-') or change.startswith('+')]
    
    # output:
    additions = []
    deletions = []
    
    # add all monograms that indicate change
    for change in changes:
        type_of_change  = 'addition' if change[0] == '+' else 'deletion'
        
        # remove unwanted symbols
        actual_change = change[2:]
        
        if type_of_change == 'addition':
            additions.append(actual_change.lower())
            
        else:
            deletions.append(actual_change)
    
    
    return additions, deletions


def find_additions_deletions_ngrams(mono_additions, mono_deletions, a, b, ngram):
    
    # split earlier version into ngrams 
    a_ngrams = list(nltk.ngrams(a.lower().split(), ngram))

    # split later version into ngrams
    b_ngrams = list(nltk.ngrams(b.lower().split(), ngram))

    # split monogram additions into ngram addtions if they appear next
    # to one onther in string b
    ngram_additions = list(nltk.ngrams(mono_additions, ngram))

    ngram_deletions = list(nltk.ngrams(mono_deletions, ngram))

    # list of additions of current ngram length
    # i.e: for ngram=2 this list only contains bigrams
    additions = []
    
    for ngram_addition in ngram_additions:
        
        # check if the two additions appear next to each other in string b
        if ngram_addition in b_ngrams:
            additions.append(" ".join(ngram_addition))
            
    deletions = []
    
    for ngram_deletion in ngram_deletions:
        
        # check if the two additions appear next to each other in string b
        if ngram_deletion in a_ngrams:
            deletions.append(" ".join(ngram_deletion))

    return additions, deletions

def find_additions_deletions_max_ngram(a, b, max_ngram, symbols_to_remove):
    
    a = utilities.remove_punctuation(a[:-1], [","])
    b = utilities.remove_punctuation(b[:-1], [","])
    
    # extract all single word additions
    mono_additions, mono_deletions = find_additions_deletions(a, b)
    
    # total list of additions
    # i.e.: for max_ngram = 3, this list contains
    # mono-, bi- and trigrams
    additions = mono_additions.copy()
    deletions = mono_deletions.copy()

    for ngram_size in range(2, max_ngram+1):
        
        # find (ngram_size)-gram additions and deletions
        current_additions, current_deletions = find_additions_deletions_ngrams(mono_additions,
                                                                            mono_deletions,
                                                                            a,
                                                                            b,
                                                                            ngram_size)

        additions += current_additions
        deletions += current_deletions
    
    
    return additions, deletions


def match_sentences_tfidf_weighted(document_a, document_b, threshold = 0.6, *args):
    
    # Use the sentences in A as queries
    queries = nltk.sent_tokenize(document_a)
    
    # Use the sentences in B as our corpus
    corpus = nltk.sent_tokenize(document_b)
    
    deleted_sentences = []
    # matched_sentences dict:
    # key = query_idx
    # value = list of matched sentences and score pairs = [(matched_sentence, similarity_score)]
    matched_sentences = {i:[(-1, 0)] for i in range(len(queries))}
    
    for query_idx in range(len(queries)):
        
        query = queries[query_idx]
        # combine the sentences into a single list
        sents = [query] + corpus

        # create a tf-idf matrix for the sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sents)

        # find the cosine similarity of the sentences in the tf-idf matrix
        cosine_similarities = np.dot(tfidf_matrix, tfidf_matrix.T)

        # print the sentences with high tf-idf weight and cosine similarity
        
        maximum_score = cosine_similarities[0, 1]
        max_idx = 0
        
        for i in range(2, len(sents)):
            if cosine_similarities[0, i] > maximum_score:
                maximum_score = cosine_similarities[0, i]
                max_idx = i - 1
                
        #print("score: ", round(maximum_score, 5), "\n")
        #print("Query:", query)
        #print("Matched:", sents[max_idx], "\n")
        if maximum_score > threshold:
            matched_sentences[query_idx] = [(max_idx, round(maximum_score, 5))]
        
        if max_idx == -1:
            deleted_sentences.append(query_idx)

    return matched_sentences, deleted_sentences



def match_sentences_semantic_search(document_a, document_b, threshold=0.6, k = 3, model='all-MiniLM-L6-v2'):
    
    # Model to be used to create Embeddings which we will use for semantic search
    embedder = SentenceTransformer(model)
    
    # Use the sentences in A as queries
    queries = nltk.sent_tokenize(document_a)
    
    # Use the sentences in B as our corpus
    corpus = nltk.sent_tokenize(document_b)
    
    # Create embeddings using B
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # list of deleted sentences from A
    deleted_sentences = []

    # matched_sentences dict:
    # key = query_idx
    # value = list of matched sentences and score pairs = [(matched_sentence, similarity_score)]
    matched_sentences = {i:[] for i in range(len(queries))}
    
    # Find the closest k most similar sentences using cosine similarity
    top_k = min(k, len(corpus))
    for query_idx in range(len(queries)):
        query = queries[query_idx]
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest k scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        # loop over top results
        match_found = False
        for score, idx in zip(top_results[0], top_results[1]):
            
            # fill the matched sentences dictonary
            if score > threshold:
                matched_sentences[query_idx].append((idx, score))
                match_found = True
        
        if not match_found:
            deleted_sentences.append(query_idx)

    return matched_sentences, deleted_sentences

def find_added_indices(matched_indices, corpus_length):
    corpus_indices = list(range(corpus_length))
    return list(set(corpus_indices) - set(matched_indices))

"""
    for i in changed_content:
        matches = list(changed_content[i].keys())
        for j in matches:
            for k in matches:
                if k != j:
                    for l in range(len(changed_content[i][j])):
                        element = changed_content[i][j][l]
                        if bool(element not in changed_content[i][k]):                            
"""

def unified_diffs(siblings):
    first = siblings[0]
    unified_content = []
    for i in range(len(first)):
        word = first[i]
        if all([word in sibling for sibling in siblings[1:]]):
            unified_content.append(word)
    
    return unified_content

def detect_changes(matched_dict, document_a, document_b, important_indices, max_ngram,top_k=1, show_output=False,       
                   symbols_to_remove=[","]):
        
    # Use the sentences in A as queries
    queries = nltk.sent_tokenize(document_a)
    
    changed_sentences = []
    
    corpus = nltk.sent_tokenize(document_b)
    
    matched_indices = []
    
    # can map to multiple, since spltis are possible
    save_additions = {i:{} for i in list(matched_dict.keys())}
    
    save_deletions = {i:{} for i in list(matched_dict.keys())}

    for query_idx in range(len(queries)):
        # current query
        query = queries[query_idx]
        
        # give lower bound on number of matched sentences
        bounded_k = min(top_k, len(matched_dict[query_idx]))
        
        for k in range(bounded_k):
            
            
            # get current matched_sentence + score
            matched_idx, score = matched_dict[query_idx][k]

            # non-matches are indicated by 0
            if matched_idx >= 0:
                matched_indices.append(int(matched_idx))

                # get the actual sentence
                matched_sentence = corpus[int(matched_idx)]


                additions, deletions = find_additions_deletions_max_ngram(query, matched_sentence, max_ngram, symbols_to_remove)

                # get syntactic ratio
                ratio = syntactic_ratio(query, matched_sentence)

                if show_output:
                    print(f"query: {query}\nmatched: {matched_sentence}\nSemantic Resemblence: {score:.4f}\n"
                          f"Syntactic Resemblence: {ratio:.4f}\n")

                    # extract addtions and deletions


                    print(f"added in newer version:{additions}\ndeleted from older version: {deletions}")

                    print("------------------------------------------------------------------------------\n")

                if ratio < 1.0:
                    changed_sentences.append(query_idx)

                    save_additions[query_idx][int(matched_idx)] = additions

                    save_deletions[query_idx][int(matched_idx)] =  deletions
                
    #drop_unimportant_indices(changed_sentences, important_indices=important_indices[version])

    new_sentences = find_added_indices(matched_indices, len(corpus))
    unified_delitions = {i: [] for i in changed_sentences}
    
    for changed_idx in changed_sentences:

        matches = list(save_deletions[changed_idx].values())
        unified_delitions[changed_idx] = unified_diffs(matches)

    return changed_sentences, new_sentences, save_additions, save_deletions, matched_indices, unified_delitions
                    
            
                    
def calculate_change_importance(changed_idx, matched_dict, ranking,
                                threshold, version,w0, w1, w2, top_k =1):
    
    """Calculate how important an actual change was.
    
    Using Hyp 1-2

    Args:
       matched_dict: calcualted using match_sentences
       ranking: sentence_importance obtained using rank_YAKE!
       threshold: Hyperparameter to determine if sentence is new
       w0-w2: optionally tuned weights
       top_k: only usefull if sentence matching is being ran with extra k
        
    Returns:
        Importance of 1 change.
    """
    
    I_s = ranking[version][changed_idx]
    
    for k in range(len(matched_dict[changed_idx])):
        matched_idx, score = matched_dict[changed_idx][k]

        next_I_s = ranking[version + 1][int(matched_idx)]

        if score < threshold:
            # Hypothesis 2
            print("A")
            I_c = next_I_s * (w2/ w1 * score)
        else:    
            # Hypothesis 1
            print("B")
            I_c = I_s * (w0/ w1 * score)
    
    return I_c


def calculate_change_importances(changed_indices, matched_dict,
                                 ranking, threshold, version, w0=1, w1=1, w2=1, k=1):
    
    return {changed_index: calculate_change_importance(changed_index, matched_dict,ranking,threshold, version, w0, w1, w2, top_k=k) 
            for changed_index in changed_indices}







