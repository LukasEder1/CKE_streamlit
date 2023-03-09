
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import yake
from summa import summarizer
import nltk
from sentence_transformers import SentenceTransformer, util
import re
from yake.highlight import TextHighlighter
import keyword_extraction
import math
import pysbd
import numpy

import nltk

from sumy.summarizers import AbstractSummarizer


class TextRankSummarizer(AbstractSummarizer):
    """An implementation of TextRank algorithm for summarization.
    Source: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    """
    epsilon = 1e-4
    damping = 0.85
    # small number to prevent zero-division error, see https://github.com/miso-belica/sumy/issues/112
    _ZERO_DIVISION_PREVENTION = 1e-7
    _stop_words = frozenset()
    seg = pysbd.Segmenter(language="en", clean=False)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        

        ratings = self.rate_sentences(document)
        
        sentences = self.seg.segment(document)
        
        return self._get_best_sentences(sentences, sentences_count, ratings)


    def rate_sentences(self, document):
        
        sentences = self.seg.segment(document)
        
        matrix = self._create_matrix(document)
        
        ranks = self.power_method(matrix, self.epsilon)
        
        # key -> sentence position, value: TextRank score
        ranking = {idx: rank for idx, rank in enumerate(ranks)}
        
        # sort the ranking by sentence importance
        return {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], 
                                                 reverse=True)}
        

    def _create_matrix(self, document):
        """Create a stochastic matrix for TextRank.
        Element at row i and column j of the matrix corresponds to the similarity of sentence i
        and j, where the similarity is computed as the number of common words between them, divided
        by their sum of logarithm of their lengths. After such matrix is created, it is turned into
        a stochastic matrix by normalizing over columns i.e. making the columns sum to one. TextRank
        uses PageRank algorithm with damping, so a damping factor is incorporated as explained in
        TextRank's paper. The resulting matrix is a stochastic matrix ready for power method.
        """
        
        sentences = self.seg.segment(document)
        
        sentences_as_words = [self._to_words_set(sent) for sent in sentences] # -> USE NLTK isntead -> easy fix
       
        sentences_count = len(sentences_as_words)
        weights = numpy.zeros((sentences_count, sentences_count))

        for i, words_i in enumerate(sentences_as_words):
            for j in range(i, sentences_count):
                rating = self._rate_sentences_edge(words_i, sentences_as_words[j])
                weights[i, j] = rating
                weights[j, i] = rating

        weights /= (weights.sum(axis=1)[:, numpy.newaxis] + self._ZERO_DIVISION_PREVENTION)

        # In the original paper, the probability of randomly moving to any of the vertices
        # is NOT divided by the number of vertices. Here we do divide it so that the power
        # method works; without this division, the stationary probability blows up. This
        # should not affect the ranking of the vertices so we can use the resulting stationary
        # probability as is without any postprocessing.
        return numpy.full((sentences_count, sentences_count), (1.-self.damping) / sentences_count) \
            + self.damping * weights

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, nltk.word_tokenize(sentence))
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    @staticmethod
    def _rate_sentences_edge(words1, words2):
        rank = sum(words2.count(w) for w in words1)
        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        if numpy.isclose(norm, 0.):
            # This should only happen when words1 and words2 only have a single word.
            # Thus, rank can only be 0 or 1.
            assert rank in (0, 1)
            return float(rank)
        else:
            return rank / norm

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector
    

def text_rank_importance(documents):
    
    versions = len(documents)
    
    summarizer = TextRankSummarizer()
    
    ranking = {i: summarizer.rate_sentences(documents[i]) for i in range(versions)}
    
    return ranking


def find_important_indices(important_sentences, document):
    seg = pysbd.Segmenter(language="en", clean=False)
    # split into sentences
    sentences = seg.segment(document)
    
    # list of indices corresponding to the important sentences 
    important_indices = []
    
    for important_sentence in important_sentences:
        for idx in range(len(sentences)):
            if important_sentence == sentences[idx]:
                important_indices.append(idx)
                
    return important_indices

def drop_unimportant_indices(indices, important_indices):
    
    # check if indices are included in the important indices of version
    return list(set(indices) & set(important_indices))

def text_rank(documents):
    
    number_of_documents = len(documents)
    important_sentences = {version:[] for version in range(number_of_documents)}
    important_indices = {version:[] for version in range(number_of_documents)}
    seg = pysbd.Segmenter(language="en", clean=False)

    for current in range(number_of_documents):
        
        # current document
        document = documents[current]
        
        # find most important sentences
        summary = summarizer.summarize(document).strip()
        
        # tokenize into seprate sentences
        important_sentences[current] = seg.segment(summary)
            
        # find corresponding indices in original corpus
        important_indices[current] = find_important_indices(important_sentences[current], document)
        
    return important_sentences, important_indices


def yake_keyword_frequency(documents, ngram_size=3):
    
    number_of_documents = len(documents)
    seg = pysbd.Segmenter(language="en", clean=False)
    # dictonary of form: version -> keywords
    keywords = keyword_extraction.extract_yake(documents)
    
    # init highlighter object
    highlighter = TextHighlighter(max_ngram_size = ngram_size)
    
    # save the sentence specific keyword counts for all documents
    keyword_counts = []
    
    for current in range(number_of_documents):
        
        # current number of sentences
        number_of_sentences = len(seg.segment(documents[current]))
        
        # dictonary to count number of keywords in a sentence
        keyword_count = {sentence:0 for sentence in range(number_of_sentences)}
        
        # highlight keywords in text
        highlightet_text = highlighter.highlight(documents[current], keywords[current])
        
        # split document into sentences
        highlightet_sentences = seg.segment(highlightet_text)
        
        # used to determine the current position in the corpus
        sentence_position = 0
        
        for sentence in highlightet_sentences:
            
            # find all keywords in current sentence
            current_keywords = re.findall(r"<kw>(.*?)</kw>", sentence)
            
            keyword_count[sentence_position] = len(current_keywords)
            
            sentence_position += 1
        
        keyword_counts.append(keyword_count)
        
    return keyword_counts


def yake_weighted_keyword_frequency(documents, ngram_size=3):
    
    number_of_documents = len(documents)
    seg = pysbd.Segmenter(language="en", clean=False)

    # dictonary of form: version -> keywords
    keywords = keyword_extraction.extract_yake(documents)
    
    # init highlighter object
    highlighter = TextHighlighter(max_ngram_size = ngram_size)
    
    # save the sentence specific keyword counts for all documents
    keyword_counts = []
    
    for current in range(number_of_documents):
        
        # current number of sentences
        number_of_sentences = len(seg.segment(documents[current]))
        
        # dictonary to count number of keywords in a sentence
        keyword_count = {sentence:0 for sentence in range(number_of_sentences)}
        
        # highlight keywords in text
        highlightet_text = highlighter.highlight(documents[current], keywords[current])
        
        # split document into sentences
        highlightet_sentences = seg.segment(highlightet_text)
        
        # used to determine the current position in the corpus
        sentence_position = 0
        
        for sentence in highlightet_sentences:
            
            # find all keywords in current sentence
            current_keywords = re.findall(r"<kw>(.*?)</kw>", sentence)
            
            # combine the score of the keywords
            # take the inverse since YAKE! goes from lowest -> highest
            combined_keyword_scores = sum([1/score for keyword, score in keywords[current] 
                                           if keyword in current_keywords])
            
            # weighted count
            keyword_count[sentence_position] = len(current_keywords) * combined_keyword_scores
            
            sentence_position += 1
        
        keyword_counts.append(keyword_count)
        
    return keyword_counts


def rank_yake(documents, keyword_counts, top_n=5):
    number_of_documents = len(documents)
    
    important_sentences = {version:[] for version in range(number_of_documents)}
    important_indices = {version:[] for version in range(number_of_documents)}
    ranking = {version:{} for version in range(number_of_documents)}
    seg = pysbd.Segmenter(language="en", clean=False)
    # document index
    current = 0
    eps = 0.001
    
    for counts in keyword_counts:
            
        sentences  = seg.segment(documents[current])
        
        # get the length of each sentence
        sentence_lengths = [len(sentence) for sentence in sentences]
        
        # normalize keyword counts by sentence length
        # In order to not give longer sentences more importance
        # + eps -> to latter avoide divison by zero
        normalized_counts = {k: (v/sentence_lengths[i]) + eps for i, (k, v) in enumerate(counts.items())}
    
        # sort counts by number of keyword frequency (highest -> lowest)
        sorted_counts = {k: v for k, v in sorted(normalized_counts.items(), key=lambda item: item[1], 
                                                 reverse=True)}
        
        # sorted sentence idices, by highest keyword frequencies
        top_keys = list(sorted_counts.keys())
        
        # return the top_n most important sentences
        important_sentences[current] = [sentences[idx] for idx in top_keys[:top_n]]
        
        # return the top_n most important indices
        important_indices[current] = top_keys[:top_n]
        
        ranking[current] = {k: v / sum(sorted_counts.values()) for k, v in sorted_counts.items()}

        current += 1
    
    return important_sentences, important_indices, ranking


def yake_weighted_importance(documents):
    #maybe inculde ngram latter
    weighted_keyword_counts = yake_weighted_keyword_frequency(documents, ngram_size=3)
    
    sentences, imp_indices, ranking = rank_yake(documents, weighted_keyword_counts)
    
    return ranking


def yake_unweighted_importance(documents):
    
    keyword_counts = yake_keyword_frequency(documents, ngram_size=3)
    
    sentences, imp_indices, ranking = rank_yake(documents, keyword_counts)
    
    return ranking

def ls(important, to_rank):
    index = 0.01
    for idx in range(len(important)):
        if important[idx] == to_rank:
            index += idx
    return index


def contrastive_importance(former, later):

    # Concat the two document versions
    combined = text_rank_importance([former + "\n" + later])[0]
    
    # reverse the values
    combined = {k: 1/v for k, v in combined.items()}
    
    total = sum(combined.values())
    
    # normalize and sort
    combined = {k: float(v/total) for k, v in sorted(combined.items(), key=lambda item: item[1], 
                                                 reverse=True)}

    # Sentence Boundary Detector (Improved Sentence Tokenization)
    # possible alternative would be nltk.sent_tokenize                                    
    seg = pysbd.Segmenter(language="en", clean=False)

    # calculate number of sentences in the earlier sentence
    former_length = len(seg.segment(former))

    # dictonaries for the respective documents
    ci_former = {k: v for k, v in combined.items() if int(k) < former_length}

    ci_later = {k-former_length: v for k, v in combined.items() if int(k) >= former_length}


    return ci_former, ci_later
