import string
import nltk
import pysbd
from collections import Counter
import itertools
import re



def get_ngrams_of_size_n(diff_content, ngram_size):
    return [ngram.lower() for ngram in diff_content if len(ngram.split(" ")) == ngram_size]


def independent_words_pairs(list):
    word_counts = Counter(get_ngrams_of_size_n(list, 1))
    words = get_ngrams_of_size_n(list, 1)
    bigrams = get_ngrams_of_size_n(list, 2)

    for word1, word2 in itertools.combinations(words, 2):
        if word1 + " " + word2  in bigrams or word2 + " " + word1 in bigrams:
            if word_counts.get(word1, 0) > 0 and word_counts.get(word2, 0) > 0:
                word_counts[word1] -= 1
                words.remove(word1)
                word_counts[word2] -= 1
                words.remove(word2)


    return words

def merge(ngram1, ngram2):
    words1 = ngram1.split(" ")
    words2 = ngram2.split(" ")

    if words1[0] == words2[-1]:
        return " ".join(words2[:-1] + words1)

    if words1[-1] == words2[0]:
        return " ".join(words1[:-1] + words2)

    return ""


def independent_ngram_pairs(list, n):
    ngrams =  get_ngrams_of_size_n(list, n)
    mgrams =  get_ngrams_of_size_n(list, n+1)
    freq = Counter(list)
    for ngram1, ngram2 in itertools.combinations(ngrams, 2):
        mgram = merge(ngram1, ngram2)
        if mgram in mgrams:
            if freq.get(ngram1, 0) > 0 and freq.get(ngram2, 0) > 0:
                freq[ngram1] -= 1
                ngrams.remove(ngram1)
                freq[ngram2] -= 1
                ngrams.remove(ngram2)
    return ngrams

def remove_lower_ngrams(list, max_ngram):
    filtered_list = []

    filtered_list += independent_words_pairs(list)
    for i in range(2, max_ngram+1):
        filtered_list += independent_ngram_pairs(list, i) 

    return filtered_list


def remove_punctuation(text, symbols=string.punctuation):
    return "".join([char for char in text if char not in symbols])

def build_sentence_freqs_ngram(sentence, n, symbols_to_remove, extra_stopwords = []):
    """
    in: sentence
    
    out: dictonary of lowercased ngram frequencies without stopwords
    """
    stop_words = extra_stopwords

    freqs = {}
    
    words = nltk.word_tokenize(sentence)
    
    words = [remove_punctuation(word.lower(), symbols_to_remove) for word in words]
    
    words = [word for word in words if len(word) > 0 and word not in stop_words]
    
    for gram in nltk.ngrams(words, n):
        
        ngram = " ".join(gram)
   
        freqs[ngram] = freqs.get(ngram, 0) + 1 

        
    return freqs


def build_sentence_freqs_max_ngram(sentence, higher_ngram, lower_ngram = 1, symbols_to_remove=[","], extra_stopwords = []):
    """
    in: sentence
    
    out: dictonary of lowercased ngrams frequencies without stopwords. [up to (max_ngram)-grams]
    i.e.: for ngram_range = (1, 3), the dict contains mono-, bi- and trigrams.
    i.e.: if lower = higher -> the dict works the same as build_sentence_freqs_ngram
    """
    
    freqs = {}
    for n in range(lower_ngram, higher_ngram+1):
        freqs.update(build_sentence_freqs_ngram(sentence, n, symbols_to_remove, extra_stopwords))


    filtered = remove_lower_ngrams(list(freqs.keys()), higher_ngram)

    return {i:freqs[i] for i in filtered}


def build_doc_level_freqs(documents, maxngram, extra_stopwords = []):
    """
    in: documents
    
    out: dictonary of dictonaries of word frequencies
    """
    
    number_of_documents = len(documents)
    
    document_frequencies = {version:{} for version in range(number_of_documents)}
    stop_words =  extra_stopwords
    current_version = 0
    seg = pysbd.Segmenter(language="en", clean=False)
    for document in documents:
        sentences = seg.segment(document.lower())
        freqs = {}
        
        
        
        for sentence in sentences:
            words = re.findall(r"[\w']+", sentence)
            for n in range(1, maxngram+1):
                for ngram in nltk.ngrams(words, n):
                    phrase = " ".join(ngram)
                    if phrase not in stop_words and len(phrase) > 0:
                        freqs[phrase] = freqs.get(phrase, 0) + 1 
        
        document_frequencies[current_version] = freqs
        current_version += 1
        
    return document_frequencies

def build_diff_level_freqs(diff_content, symbols_to_remove, stop_words):
    """
    in: documents 
    
    out: dictonary of dictonaries of word frequencies for diff content
    """
    diff_content = [remove_punctuation(word.lower(), [","]) for word in diff_content]

    #stop_words = nltk.corpus.stopwords.words("english")

    diff_content = [word for word in diff_content if len(word) > 0 and word not in stop_words]
        
    return Counter(diff_content) 


def build_sentence_level_freqs(document):
    """
    in: documents
    
    out: dictonary of dictonaries of word frequencies without stopwords
    """
    
    seg = pysbd.Segmenter(language="en", clean=False)

    sentences = seg.segment(document)
      
    sentences = [sentence.lower() for sentence in sentences]
    
    number_of_sentences = len(sentences)
    
    
    
    sentence_frequencies = {version:{} for version in range(number_of_sentences)}
    
    current_version = 0

   
    stop_words = nltk.corpus.stopwords.words("english")
    
    for sentence in sentences:
        freqs = {}
        for word in nltk.word_tokenize(sentence):
            word = remove_punctuation(word)
            if word not in stop_words and len(word) > 0:
                freqs[word] = freqs.get(word, 0) + 1 

        sentence_frequencies[current_version] = freqs
        current_version += 1
        
    return sentence_frequencies


def build_sentence_freqs(sentence, symbols_to_remove):
    """
    in: sentence
    
    out: dictonary of word frequencies without stopwords
    """

    stop_words = nltk.corpus.stopwords.words("english")
    
    freqs = {}
    
    for word in nltk.word_tokenize(sentence.lower()):
        
        # remove unwanted symbols
        word = remove_punctuation(word, symbols_to_remove)
        
        # len(word) > 0, needed in case word is a punctuation
        if word not in stop_words and len(word) > 0:
            freqs[word] = freqs.get(word, 0) + 1 

        
    return freqs

def drop_unimportant_indices(indices, important_indices):
    
    # of some version!
    return list(set(indices) & set(important_indices))

def alpha_combination(I_c, I_s, alpha=0.5):
    return alpha * I_s + (1- alpha) * I_c

def gamma_combination(I_c, I_s, gamma=0.5):
    return I_s ** gamma * I_c ** (1- gamma)

def harmonic_mean(I_c, I_s, gamma):
    return (2 * I_c * I_s) / (I_c + I_s)

def normalize_list(lst):
    total_sum = sum(lst)
    normalized_values = [value / total_sum for value in lst]
    return normalized_values