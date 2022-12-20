import string
import nltk
import utilities
from tqdm import trange
import sentence_importance
from sentence_comparision import calculate_change_importances, match_sentences_tfidf_weighted, detect_changes, match_sentences_semantic_search

def normalize_scores(keywords):

    if len(keywords) == 0:

        return {}

    max_value = max([item[1] for item in keywords])

    min_value = min([item[1] for item in keywords])
    
    print(max_value)
    
    print(min_value)
   

    result = {}

    for item in keywords:
        
        print(item)
        normalized_score = (item[1] - float(min_value))/(float(max_value) - float(min_value))

        result[item[0]] = abs(1 - normalized_score)

 

    return result


def final_score(document, version, changed_indices, new_indices, matched_dict, ranking, I_c, max_ngram, 
                additions, combinator=utilities.alpha_combination, k=0, alpha_gamma=0.5, min_ngram = 1,
                symbols_to_remove=string.punctuation):
    
    
    
    # tokenize document into sentencs
    sentences = nltk.sent_tokenize(document) 
    
    
    # Importance of sentences for current document
    I_s = ranking[version]
    

    # computed intermediate Keywords for contrastive KE between the current and prev Document Version    
    keywords = {}
    
    # loop over the changed sentences
    for i in changed_indices:
        
        # extract the correspdoning matched sentence
        matched_idx, score = matched_dict[i][k]
        
        # Retrieve the Importance of the change that led to new sentence
        I_ci = I_c[i]
        
        # Retrieve the Importance of the matched sentence
        I_si = I_s[int(matched_idx)] 
        
        # Combine the two scores using a combinator
        s_c = combinator(I_ci, I_si, alpha_gamma)
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in matched sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences[matched_idx], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove)
        
        
        # loop over all ngrams/freqs in the sentence
        for ngram, freq in current_freqs.items():
            
            # check if word/ngram has been newly added (check the find_addition_deletions function for details)
            if ngram in additions[i]:
                
                # include added ngrams, scored by their frequency * score of the change 
                keywords[ngram] = keywords.get(ngram, 0) + float(freq * s_c)
            

                    
    # newly added sentence: ( new := has not been matched to)
    for i in new_indices:
        
        # Compute the Dictonary of frequency for all ngrams up to "max_ngram" in new sentence
        current_freqs = utilities.build_sentence_freqs_max_ngram(sentences[i], 
                                                       higher_ngram=max_ngram, lower_ngram=min_ngram,
                                                       symbols_to_remove=symbols_to_remove)
        
        
        for ngram, freq in current_freqs.items():
            
            # include added ngrams, scored by their frequency * Importance of the sentence
            keywords[ngram] = keywords.get(ngram, 0) + float(freq * I_s[i])
            
   
    # normalize keywords
    # total "IMPORTANCE COUNT
    total_count = sum(keywords.values())
    
    # sort keywords + normalize
    keywords = {k: v/total_count  for k, v in sorted(keywords.items(), key=lambda item: item[1], 
                                                 reverse=True)}
    
    return keywords



def contrastive_extraction(documents, max_ngram, min_ngram=1, 
                           importance_estimator= sentence_importance.text_rank_importance,
                           combinator=utilities.alpha_combination, threshold=0.6, top_k=1, alpha_gamma=0.5, 
                           matching_model='all-MiniLM-L6-v2', w0 = 3, w1 = 1, w2 = 1, match_sentences =match_sentences_semantic_search, show_changes=False, symbols_to_remove=[","]):
    
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
    
    for i in range(versions-1):
        
        i_next = i + 1
        
        # matching
        matched_dict = match_sentences(documents[i], documents[i+1], top_k, matching_model)
        
        
        matched_dicts[i] = matched_dict
        
        # determine WHAT has changed
        changed_indices, new_indices, adds, delet = detect_changes(matched_dict, documents[i], documents[i+1], 
                                           important_indices=[], max_ngram=max_ngram, show_output=show_changes,
                                           symbols_to_remove=symbols_to_remove)
        
        additions[i] = adds
        
        deletions[i] = delet
        
        
        changed_sentences[i] = changed_indices
        
        # determine HOW important the change was
        I_c = calculate_change_importances(changed_indices, matched_dict, ranking ,threshold, 
                                           version=i, w0 = w0, w1 = w1, w2=w2)
        
        # calculate keywords between two subsequent versions
        intermediate_keywords = final_score(documents[i+1], i+1, changed_indices, new_indices, matched_dict, 
                                            ranking, I_c, max_ngram, adds, combinator, 
                                            alpha_gamma=alpha_gamma, min_ngram= min_ngram, 
                                            symbols_to_remove=symbols_to_remove)
        
        # add to overall dictonary
        # index n: contrastive keywords for versions n and n+1
        keyword_collection[i] = intermediate_keywords
    
    return keyword_collection, matched_dicts, changed_sentences, additions, deletions





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







