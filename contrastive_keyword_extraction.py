import string
import nltk
import utilities
from tqdm import trange
import sentence_importance
from sentence_comparision import calculate_change_importances, match_sentences_tfidf_weighted, detect_changes, match_sentences_semantic_search

def normalize_scores(keywords):

    if len(keywords) == 0:

        return {}

    max_value = max([v for v in keywords.values()])
    print(max_value)
    min_value = min([v for v in keywords.values()])
    print(min_value)
    result = {}

    for key, value in keywords.items():
        
        normalized_score = (value - float(min_value))/(float(max_value) - float(min_value))

        result[key] = abs(1 - normalized_score)

 

    return result


def final_score(documents, changed_indices, new_indices, matched_dict, ranking, max_ngram, 
                additions, removed_indices, deleted, combinator=utilities.alpha_combination, top_k=0, alpha_gamma=0.5, min_ngram = 1,
                symbols_to_remove=string.punctuation, extra_stopwords=[]):
    
    
    # tokenize document into sentencs
    sentences_a = nltk.sent_tokenize(documents[0]) 
    
    # tokenize document into sentencs
    sentences_b = nltk.sent_tokenize(documents[-1]) 

    # Importance of sentences for current document
    I_sprev = ranking[0]
    I_s = ranking[1]
    

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

            # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in matched sentence
            current_freqs = utilities.build_sentence_freqs_max_ngram(sentences_b[matched_idx], 
                                                        higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                        symbols_to_remove=symbols_to_remove,
                                                        extra_stopwords=extra_stopwords)
            
            # loop over all ngrams/freqs in the sentence
            for ngram, freq in current_freqs.items():
                
                # check if word/ngram has been newly added (check the find_addition_deletions function for details)
                if ngram in additions[i][int(matched_idx)]:
                    # include added ngrams, scored by their frequency * score of the change 
                    keywords[ngram] = keywords.get(ngram, 0) + float(freq * s_c)
                
            
        # get frequencies of sentence in older version
        # in order to include deletions as keywords
        old_freqs = utilities.build_sentence_freqs_max_ngram(sentences_a[i], 
                                                        higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                        symbols_to_remove=symbols_to_remove,
                                                        extra_stopwords=extra_stopwords)
        
        # loop over all ngrams/freqs in the sentence
        for ngram, freq in old_freqs.items():
            
            # check if word/ngram has been deleted from old
            # (check the find_addition_deletions function for details)
            if ngram in deleted[i]:
                # include deleted ngrams, scored by their frequency * score of the change 
                keywords[ngram] = keywords.get(ngram, 0) + float(freq * s_c)


    # newly added sentence: ( new := has not been matched to)
    for i in new_indices:
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in new sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences_b[i], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove,
                                                       extra_stopwords=extra_stopwords)
        
        
        for ngram, freq in current_freqs.items():
            
            # include added ngrams, scored by their frequency * Importance of the sentence
            keywords[ngram] = keywords.get(ngram, 0) + float(freq * I_s[i])
            
    
    # removed sentence: ( removed := no match found)
    for i in removed_indices:
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in deleted sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences_a[i], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove,
                                                       extra_stopwords=extra_stopwords)
        
        
        for ngram, freq in current_freqs.items():
            
            # include deleted ngrams, scored by their frequency * Importance of the sentence
            keywords[ngram] = keywords.get(ngram, 0) + float(freq * I_sprev[i])

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
    
    versions = len(documents)
    
    
    #rank all sentences in their respective version in the total document catalogue
    # available esitmators: text_rank_importance, yake_weighted_importance, yake_unweighted_importance 
    ranking = importance_estimator(documents)
    
    # intermediate keywords
    keyword_collection = {version:{} for version in range(versions-1)}
    
    changed_sentences = {version: [] for version in range(versions-1)}
    
    matched_dicts = {version: {} for version in range(versions-1)}
    
    additions = {version: {} for version in range(versions-1)}
    
    deletions = {version: {} for version in range(versions-1)}

    new = {version: [] for version in range(versions-1)}

    removed = {version: [] for version in range(versions-1)}

    for i in range(versions-1):
        
        i_next = i + 1
        
        # matching
        matched_dict, rmv = match_sentences(documents[i], documents[i+1],threshold, top_k, matching_model)
        
        
        matched_dicts[i] = matched_dict
        
        # determine WHAT has changed
        changed_indices, new_indices, adds, delet, matched_indices, unified_delitions = detect_changes(matched_dict, documents[i], documents[i+1], 
                                           important_indices=[], max_ngram=max_ngram, show_output=show_changes,
                                           symbols_to_remove=symbols_to_remove, top_k=top_k)
        
        
        additions[i] = adds
        
        deletions[i] = delet

        new[i] = new_indices
        
        removed[i] = rmv
        
        changed_sentences[i] = changed_indices
        
        # calculate keywords between two subsequent versions
        intermediate_keywords = final_score(documents, changed_indices, new_indices, matched_dict, 
                                            ranking, max_ngram, adds, rmv, unified_delitions, combinator, 
                                            alpha_gamma=alpha_gamma, min_ngram= min_ngram, 
                                            symbols_to_remove=symbols_to_remove,
                                            extra_stopwords=extra_stopwords)
        
        # add to overall dictonary
        # index n: contrastive keywords for versions n and n+1
        keyword_collection[i] = intermediate_keywords
    
    return keyword_collection, matched_dicts, changed_sentences, additions, deletions, new, ranking, removed, matched_indices, unified_delitions





def combine_keywords(keywords):
    total_keywords = {}
    
    # normalize keyword values
    normalization_term = len(keywords)
    
    for idx in keywords:
        current_keywords = keywords[idx]
        
        for keyword, value in current_keywords.items():
            
            # sum up all keywords in the different versions
            total_keywords[keyword] = total_keywords.get(keyword, 0) + (value / normalization_term)
    
    # sorted the keywords
    sorted_keywords = {k: v for k, v in sorted(total_keywords.items(), key=lambda item: item[1], 
                                            reverse=True)}
    
    return sorted_keywords







