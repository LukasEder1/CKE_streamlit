import string
import nltk
import utilities
from tqdm import trange
import sentence_importance
from sentence_comparision import *
import pysbd
from collections import Counter
from utilities import * 
import numpy as np

def final_score(documents, changed_indices, new_indices, matched_dict, ranking, max_ngram,
                additions, removed_indices, deleted, combinator=utilities.alpha_combination, 
                top_k=0, alpha_gamma=0.5, min_ngram = 1,
                symbols_to_remove=string.punctuation, extra_stopwords=[]):
    
    # init Sentence Boundary Detector
    seg = pysbd.Segmenter(language="en", clean=False)

    # tokenize document into sentencs
    sentences_a = seg.segment(documents[0]) 
    
    # tokenize document into sentencs
    sentences_b = seg.segment(documents[-1]) 

    doc_level_stats = build_doc_level_freqs(documents, maxngram=max_ngram, extra_stopwords=extra_stopwords)

    # Importance of sentences for current document
    I_sprev = ranking[0]
    I_s = ranking[1]
    
    former_contrastiveness, latter_contrastiveness = sentence_importance.contrastive_importance(documents[0], documents[-1])

    # computed intermediate Keywords for contrastive KE between the current and prev Document Version    
    keywords = {}
    
    # loop over the changed sentences
    for i in list(set(changed_indices)):
        for k in range(len(matched_dict[i])):
            # extract the correspdoning matched sentence
            matched_idx, score = matched_dict[i][k]
            
            # Calculate the Importance of the change that led to new sentence
            # Hypothesis 1: Importance of previous Sentence times semantic similarity
            # of changed and matched sentence
            I_ci = I_sprev[i] * score

            # Retrieve the Importance of the matched sentence
            I_si = I_s[int(matched_idx)] 
            
            # Combine the two scores using a combinator
            s_c = combinator(I_ci, I_si, alpha_gamma)

            current_adds = additions[i].get(int(matched_idx), [])
            current_freqs = build_diff_level_freqs(current_adds, symbols_to_remove)
            # loop over all ngrams/freqs in the sentence
            
            for ngram, freq in current_freqs.items():
                # ratio := fl / fe 
                # fe ... frequency of ngram in earlier version
                # fl ... frequency of ngram in latter version
                ratio = float(doc_level_stats[1].get(ngram, 1)) / float(doc_level_stats[0].get(ngram, 1))

                # include added ngrams, scored by their frequency * score of the change 
                keywords[ngram] = keywords.get(ngram, 0) + float(ratio * freq * s_c)
                
        # get frequencies of sentence in older version
        # in order to include deletions as keywords
        # incase of presence of splits: deleted = unified_diff
        current_deletions = deleted[i]
        old_freqs = build_diff_level_freqs(current_deletions, symbols_to_remove)

        # loop over all ngrams/freqs in the sentence
        for ngram, freq in old_freqs.items():
            # ratio := fe / fl
            # fe ... frequency of ngram in earlier version
            # fl ... frequency of ngram in latter version
            ratio = doc_level_stats[0][ngram] / doc_level_stats[1].get(ngram, 1)
            
            # include deleted ngrams, scored by their frequency * score of the change
            keywords[ngram] = keywords.get(ngram, 0) + float(ratio * freq * s_c)


    # newly added sentence: ( new := has not been matched to)
    for i in new_indices:
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in new sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences_b[i], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove,
                                                       extra_stopwords=extra_stopwords)
        
        
        for ngram, freq in current_freqs.items():
            
            ratio = float(doc_level_stats[1].get(ngram, 1)) / float(doc_level_stats[0].get(ngram, 1))
            
            # include added ngrams, scored by their frequency * Importance of the sentence
            keywords[ngram] = keywords.get(ngram, 0) + float(freq * latter_contrastiveness[i] * ratio)
            
    
    # removed sentence: ( removed := no match found)
    for i in removed_indices:
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in deleted sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences_a[i], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove,
                                                       extra_stopwords=extra_stopwords)
        
        
        for ngram, freq in current_freqs.items():
            
            ratio = float(doc_level_stats[0].get(ngram, 1)) / float(doc_level_stats[1].get(ngram, 1))

            # include deleted ngrams, scored by their frequency * Importance of the sentence
            keywords[ngram] = keywords.get(ngram, 0) + float(ratio * freq * former_contrastiveness[i])

    # normalize keywords
    # total "IMPORTANCE COUNT
    total_count = sum(keywords.values())
    
    # sort keywords + normalize
    keywords = {k: float(v)/float(total_count)  for k, v in sorted(keywords.items(), key=lambda item: item[1], 
                                                 reverse=True)}
    
    return keywords



def contrastive_extraction(documents, max_ngram, min_ngram=1, 
                           importance_estimator= sentence_importance.text_rank_importance,
                           combinator=utilities.alpha_combination, threshold=0.6, top_k=1, alpha_gamma=0.5, 
                           matching_model='all-MiniLM-L6-v2', w0 = 3, w1 = 1, w2 = 1, 
                           match_sentences =match_sentences_semantic_search, show_changes=False,
                           symbols_to_remove=[","], extra_stopwords=[]):
    
    
    # rank all sentences in their respective version
    # available esitmators: text_rank_importance, yake_weighted_importance, yake_unweighted_importance 
    ranking = importance_estimator(documents)
    
    # Perform Matching
    # matchers: semantic_search, weighted_tfidf
    matched_dict, removed = match_sentences(documents[0], documents[-1],threshold, top_k, matching_model)
    
    
    # determine WHAT has changed
    # Using Myers algorithm
    changed_indices, new_indices, additions, deletions, matched_indices, unified_delitions = detect_changes(matched_dict, documents[0], documents[-1], 
                                        important_indices=[], max_ngram=max_ngram, show_output=show_changes,
                                        symbols_to_remove=symbols_to_remove, top_k=top_k,
                                        extra_stopwords=extra_stopwords)
    
    

    # extract keywords
    keywords = final_score(documents, changed_indices, new_indices, matched_dict, 
                                        ranking, max_ngram, additions, removed,
                                        unified_delitions, combinator, 
                                        alpha_gamma=alpha_gamma, min_ngram= min_ngram, 
                                        symbols_to_remove=symbols_to_remove,
                                        extra_stopwords=extra_stopwords)
    
    
    return keywords, matched_dict, changed_indices, additions, deletions, new_indices, ranking, removed, matched_indices, unified_delitions