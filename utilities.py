import string
import nltk

def remove_punctuation(text, symbols=string.punctuation):
    return "".join([char for char in text if char not in symbols])


def build_sentence_freqs_ngram(sentence, n, symbols_to_remove, extra_stopwords = []):
    """
    in: sentence
    
    out: dictonary of lowercased ngram frequencies without stopwords
    """

    stop_words = nltk.corpus.stopwords.words("english") + extra_stopwords
    
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

        
    return freqs


def build_doc_level_freqs(documents):
    """
    in: documents
    
    out: dictonary of dictonaries of word frequencies
    """
    
    number_of_documents = len(documents)
    
    document_frequencies = {version:{} for version in range(number_of_documents)}
    
    current_version = 0
    
    for document in documents:
        sentences = nltk.sent_tokenize(document.lower())
        freqs = {}
        
        stop_words = nltk.corpus.stopwords.words("english")
        
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence):
                if word not in stop_words and len(word) > 0:
                    freqs[word] = freqs.get(word, 0) + 1 
        
        document_frequencies[current_version] = freqs
        current_version += 1
        
    return document_frequencies


def build_sentence_level_freqs(document):
    """
    in: documents
    
    out: dictonary of dictonaries of word frequencies without stopwords
    """
    
    sentences = nltk.sent_tokenize(document)
      
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

def get_sentences(indices, version):
    # get all sentences from version v present in list "indices"
    sentences = nltk.sent_tokenize(documents[version])
    return [sentences[index] for index in indices]


def alpha_combination(I_c, I_s, alpha=0.5):
    return alpha * I_s + (1- alpha) * I_c

def gamma_combination(I_c, I_s, gamma=0.5):
    return I_s ** gamma * I_c ** (1- gamma)

def harmonic_mean(I_c, I_s):
    return (2 * I_c * I_s) / (I_c + I_s)

