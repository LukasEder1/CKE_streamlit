from sentence_comparision import find_additions_deletions_max_ngram
import keyword_extraction


def additions_as_document(added_idx):
    """
    input: added content in current delta i.e.: delta = 0-1
    -> added contains the words added to version 1
    """
    
    document = ""
    for position, additions in added_idx.items():
        if len(additions) > 0:
            document += " ".join(additions) + ". "
            
    return document[:-1]


def baseline_diff_content(added, ke_extractor):
    
    documents = [additions_as_document(added[i]) for i in added.keys()]
    
    return ke_extractor(documents)



def create_baseline_frame(baseline_keywords):
    keywords_dict = list(baseline_keywords.values())
    kws = []
    scores = []
    delta_version = []
    for i in range(len(baseline_keywords)):
        delta_version += [i] * len(keywords_dict[i])

        for kw, score in keywords_dict[i]:
            kws.append(kw)
            scores.append(score)
    
    return kws, scores, delta_version

def create_inter_frame(inter_keywords):
    keywords_dict = list(inter_keywords.values())
    kws = []
    scores = []
    delta_version = []
    for i in range(len(inter_keywords)):
        delta_version += [i] * len(keywords_dict[i])

        for kw, score in keywords_dict[i].items():
            kws.append(kw)
            scores.append(score)
    
    return kws, scores, delta_version