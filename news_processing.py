import pandas as pd
import html2text
import re

def create_data(entry_id, connection):
    data = pd.read_sql(f"SELECT * FROM split_sentences where entry_id = {entry_id}", con=connection)
    data['sentence'] = data['sentence'].str.replace(r'<[^<>]*>', '', regex=True)
    return data

def get_versions(data):
    return set(data.version)

def get_text(data, version):
    versions = set(data.version)
    if version in versions:
        s = ""
        for sentence in data[data.version == version].sentence:
            s += sentence
            
        return s.split('--')[1]
    else:
        return ""
     
def print_version(data, version):
    versions = set(data.version)
    if version in versions:
        for sentence in data[data.version == version].sentence:
            print(sentence, "\n")
    else:
        print(f"unsupported version: {version} -  available versions: {versions}")
        
def find_valid_article_ids(ids, connection):
    articles = []
    for article_id in ids:
        if len(doc_level_stats(article_id, connection)) > 0:
            articles.append(article_id)
    return articles

def get_documents(data):
    return [get_text(data, version) for version in get_versions(data)]

def doc_level_stats(entry_id, connection):
    return pd.read_sql(f"SELECT * FROM doc_level_stats where entry_id = {entry_id}", con=connection)

def show_tables(connection):
    return (pd.read_sql("""SELECT name FROM sqlite_master WHERE type='table';""", con=connection))

def show_table(name, connection, n=5):
    return (pd.read_sql(f"SELECT * FROM {name} LIMIT {n};", con=connection))

def matched_sentences(connection, entry_id):
    return (pd.read_sql(f"SELECT * FROM matched_sentences WHERE entry_id = {entry_id};", con=connection))

def get_url(entry_id, connection):
    df = pd.read_sql(f"SELECT * FROM entry WHERE id = {entry_id};", con=connection)
    return df.url


def html_to_text(data, version, h, through_out, header_size):
    
    # split into sentences
    html_sentences = data[data.version == version]['sentence'].values
    
    # remove html_tags and newlines
    text_sentences = [re.sub('\n', ' ', h.handle(html_sentence)) for html_sentence in html_sentences]
    
    # remove header text: i.e.: 'HOUSTON (AP) -- '
    text_sentences[0] = text_sentences[0][header_size:]
    
    # through away the last line(s)
    return text_sentences[:through_out]

def create_documents(data, header_size=0, through_out=-1):
    
    # set of available versions
    versions = get_versions(data)
    
    # html to text parser
    h = html2text.HTML2Text()
    
    return {int(version): html_to_text(data, version, h, through_out, header_size) for version in versions}

