from contrastive_keyword_extraction import contrastive_extraction, final_score, combine_keywords
import sqlite3
import pandas as pd
import string
import sentence_importance
import streamlit as st
from baselines import create_inter_frame
import news_processing
import nltk
import numpy as np
import re

nltk.download("popular")

def colour_map(keywords, n):
    """
    score -> between 0 -> 1
    """
    colouring = {}
    colours = np.linspace(100, 255, n, endpoint=True)
    
    i = 0
    for kw, score in keywords.items():
        colouring[kw] = colours[i]
        i += 1
    return colouring

def create_stylesheet(keywords, colouring):
    css_string = "<style>"
    for label, score in keywords.items():
        css_string += f" b.{label} {{background-color: rgb(0,{colouring[label]},0);}}"

    css_string += " </style>"
    
    return css_string


def highlight_keywords(document, intermediate_keywords, changed_indices, matched_dict, ngram, added):
    sentences = nltk.sent_tokenize(document)
    
    g_values = colour_map(intermediate_keywords, len(intermediate_keywords))
    
    highlighted_string = ""

    for i in changed_indices:
        
        matched_idx, _ = matched_dict[i][0]

        sentence = sentences[int(matched_idx)]
        
        print("BEFORE \n", sentence)
        for keyword in intermediate_keywords.keys():
            if keyword.lower() in added[i]:
                
                sentence = re.sub(keyword.lower(), 
                    f"<b class=\"{keyword.lower()}\">" +  keyword +"</b>",
                sentence, flags=re.I)

        print("AFTER \n")
        
        highlighted_string += sentence
       
    
    
    print(highlighted_string)
    highlighted_string += create_stylesheet(intermediate_keywords, g_values)
    
    return highlighted_string


st.set_page_config(layout="wide")
st.header('Contrastive Keyword Extraction')
a = 'Porter Moser stood in front of the scarf-clad Loyola cheering section, a bit dazed but beaming from ear to ear. "Are you kidding me!Are you kidding me," the Ramblers coach screamed over and over. No kidding. Loyola is headed to the Final Four . An improbable NCAA Tournament took its craziest turn yet Saturday night, when Ben Richardson scored a career-high 23 points and the 11th-seeded Ramblers romped to a 78-62 victory over Kansas State to cap off a stunning run through the bracket-busting South Regional. The Ramblers (32-5) matched the lowest-seeded team ever to reach the Final Four, joining LSU (1986), George Mason (2006) and VCU (2011).Those other three all lost in the national semifinals. Don't bet against Loyola, which emerged from a regional that produced a staggering array of upsets.The South became the first regional in tournament history to have the top four seeds - including overall No. 1 Virginia - knocked out on the opening weekend. In the end, it was the Ramblers cutting down the nets. After three close calls, this one was downright easy. "We believed that we could do something like this - do something really special - because we knew we had such good chemistry and we've got such a good group," said Richardson, who was named MVP of the regional. "Everyone would say we were crazy.If we said this was going to happen, people would call us crazy, but you've just got to believe." No one believes more than their 98-year-old team chaplain, Sister Jean Dolores Schmidt , who led a prayer in the locker room before the game.Later, she was pushed onto the court in her wheelchair to join the celebration when it was done. Sister Jean donned a Final Four cap - she even turned it around backward, just to show she's hip to the kids - and gave a gleeful thumbs-up. She's already looking forward to a bigger game next weekend. "I'm going to San Antonio," she said. "That's going to be great." Also joining the celebration were several players from the Ramblers' 1963 national championship team , which played one of the most socially significant games in college basketball history on its way to the title.It was known as the "Game of Change," matching the Ramblers and their mostly black roster against an all-white Mississippi State team at the height of the civil rights movement, setting up an even more noteworthy contest three years later.Texas Western, with five African-American starters, defeated Kentucky in the national championship game. Les Hunter, a member of that '63 team, said these Ramblers are capable of bringing home another title. "I think they're the best right now," Hunter said. "They work so well together.They can play with anybody - anybody - right now." Even with a title on its resume, this Loyola performance came out of nowhere.The Ramblers had not made the tournament since 1985 until they broke the drought by winning the Missouri Valley Conference. Then, as if benefiting from some sort of divine intervention, the Ramblers won their first three tournament games by a total of four points . Finally, with the Final Four on the line, they turned in a thoroughly dominating performance against the ninth-seeded Wildcats (25-12), the other half of the first 9-vs.-11 matchup in tournament history. Not the least bit intimidated, Loyola came out in attack mode right from the start against a Kansas State team that rode a stifling defense to the regional final.Moving the ball just as you'd expect from a veteran squad with two seniors and two fourth-year juniors in the starting lineup, the Ramblers kept getting open looks and bolted to a 36-24 lead. "They jumped out to that big lead and it was tough for us to come back," said Xavier Sneed, who led Kansas State with 16 points."They kept their foot on the gas." The Ramblers shot 57 percent against a team that is used to shutting opponents down, including 9 of 18 from 3-point range. Kansas State hit just 35 percent from the field - 6 of 26 from beyond the arc. Early on the second half, Richardson swished a 3-pointer as he was fouled by Kamau Stokes , winding up flat on his back, flashing a huge smile with his arms raised above his head.He knocked down the free throw to complete the four-point play, stretching the lead to 44-29. Loyola led by as many as 23. "We're just a bunch of guys that everybody laughed at ...when we thought we were going to play Division I basketball," Clayton Custer said. "Nobody thought we could do any of this." They do now. BIG PICTURE Loyola: While Richardson was the top scorer, the Ramblers got contributions from everyone.Marques Townes (13 points) and Donte Ingram (12) were also in double figures, while burly freshman center Cameron Krutwig came up big on the inside (nine points, seven rebounds). Kansas State: The Wildcats were surrendering an average of 53.3 points per game in the NCAA Tournament and had not allowed more than 59 in their first three games.Loyola went by that with more than 9 minutes to go, which was more than enough to hold off the Wildcats even when things got a little sloppy in the closing minutes. UP NEXT Loyola: The Ramblers will meet the winner of the West Regional final between Florida State and Michigan next Saturday in San Antonio.They will try to become the lowest-seeded team to win a national championship, a distinction held by No. 8 seed Villanova in 1985. Kansas State: After their impressive run in the tournament, the Wildcats will face heightened expectations next season.All five starters, plus ailing forward Dean Wade, can return next season.'
b = ' Porter Moser stood in front of the scarf-clad Loyola cheering section, a bit dazed but beaming from ear to ear. "Are you kidding me!Are you kidding me!" the Ramblers coach screamed over and over. No kidding. Loyola is headed to the Final Four . An improbable NCAA Tournament took its craziest turn yet Saturday night, when Ben Richardson scored a career-high 23 points and the 11th-seeded Ramblers romped to a 78-62 victory over Kansas State to cap off a stunning run through the bracket-busting South Regional. The Ramblers (32-5) matched the lowest-seeded team ever to reach the Final Four, joining LSU (1986), George Mason (2006) and VCU (2011).Those other three all lost in the national semifinals. Don't bet against Loyola, which emerged from a regional that produced a staggering array of upsets.The South became the first regional in tournament history to have the top four seeds - including overall No. 1 Virginia - knocked out on the opening weekend. In the end, it was the Ramblers cutting down the nets. After three close calls, this one was downright easy. "We believed that we could do something like this - do something really special - because we knew we had such good chemistry and we've got such a good group," said Richardson, who was named MVP of the regional. "Everyone would say we were crazy.If we said this was going to happen, people would call us crazy, but you've just got to believe." No one believes more than their 98-year-old team chaplain, Sister Jean Dolores Schmidt , who led a prayer in the locker room before the game.When it was done, she was pushed onto the confetti-covered court in her wheelchair to join the celebration. Sister Jean donned a Final Four cap - she even turned it around backward, just to show she's hip to the kids - and gave a gleeful thumbs-up. She's already looking forward to a bigger game next weekend. "I'm going to San Antonio," she said. "That's going to be great." Also joining the celebration were several players from the Ramblers' 1963 national championship team , which played one of the most socially significant games in college basketball history on its way to the title.It was known as the "Game of Change," matching the Ramblers and their mostly black roster against an all-white Mississippi State team at the height of the civil rights movement, setting up an even more noteworthy contest three years later when Texas Western, with five African-American starters, defeated Kentucky in the national championship game. Les Hunter, a member of that '63 team, said these Ramblers are capable of bringing home another title. "I think they're the best right now," Hunter said. "They work so well together.They can play with anybody - anybody - right now." Even with a title on its resume, this Loyola performance came out of nowhere.The Ramblers had not made the tournament since 1985 until they broke the drought by winning the Missouri Valley Conference. Then, as if benefiting from some sort of divine intervention, the Ramblers won their first three tournament games by a total of four points . Finally, with the Final Four on the line, they turned in a thoroughly dominating performance against the ninth-seeded Wildcats (25-12), the other half of the first 9-vs.-11 matchup in tournament history. Not the least bit intimidated, Loyola came out in attack mode right from the start against a Kansas State team that rode a stifling defense to the regional final.Moving the ball just as you'd expect from a veteran squad with two seniors and two fourth-year juniors in the starting lineup, the Ramblers kept getting open looks and bolted to a 36-24 halftime lead. "They jumped out to that big lead and it was tough for us to come back," said Xavier Sneed, who led Kansas State with 16 points."They kept their foot on the gas." The Ramblers shot 57 percent against a team that is used to shutting opponents down, including 9 of 18 from 3-point range. Kansas State hit just 35 percent from the field - 6 of 26 beyond the arc. Early on the second half, Richardson swished a 3-pointer as he was fouled by Kamau Stokes , winding up flat on his back, flashing a huge smile with his arms raised above his head.He knocked down the free throw to complete the four-point play, stretching the lead to 44-29. Loyola led by as many as 23. "We're just a bunch of guys that everybody laughed at ...when we thought we were going to play Division I basketball," Clayton Custer said. "Nobody thought we could do any of this." They do now. BIG PICTURE Loyola: While Richardson was the top scorer, the Ramblers got contributions from everyone.Marques Townes (13 points) and Donte Ingram (12) were also in double figures, while burly freshman center Cameron Krutwig came up big on the inside (nine points, seven rebounds). Kansas State: The Wildcats were surrendering an average of 53.3 points per game in the NCAA Tournament and had not allowed more than 59 in their first three games.Loyola went by that with more than 9 minutes to go, which was more than enough to hold off the Wildcats even when things got a little sloppy in the closing minutes. UP NEXT Loyola: The Ramblers will meet West Regional winner Michigan next Saturday in San Antonio.The Wolverines beat Florida State 58-54.Loyola has a shot at becoming the lowest-seeded team to win a national championship, a distinction held by No. 8 seed Villanova since 1985. Kansas State: After their impressive run in the tournament, the Wildcats will face heightened expectations next season.All five starters, plus ailing forward Dean Wade, can return next season.'
            
#conn_news = sqlite3.connect('../datasets/ap-matched-sentences.db')
pd.set_option('display.max_columns', None)


def get_doc(article_id=17313):
    data_news = news_processing.create_data(article_id, conn_news)
            
    documents = news_processing.get_documents(data_news)

    return documents

def added_df(added):
    return pd.DataFrame({"sentence": added.keys(), "added": added.values()}).reset_index(drop=True)
#print(documents[0])



def display_keywords(keywords, k):
    inter_kws, inter_scores, delta_int = create_inter_frame(keywords)
    df = pd.DataFrame({'delta': delta_int, 'keyword': inter_kws, 'score': inter_scores})
    
    return df.head(k)

article_id = st.selectbox(
    'Choose an article',
    (17313, 17348, 16832, 17313))


col1, col2 = st.columns(2)


#documents = get_doc(article_id)

with col1:
    ngram = st.slider("Max Ngram:", 1, 6)
    former = st.text_area('Orignal Version: ', a, height=400)
    

with col2:
    top_k = st.slider("Top-k Keywords:", 5, 30)
    later = st.text_area('Later Version: ', b, height=400)
    

run = st.button('run')

if run:
    keywords, matched_dicts, changed_sentences, added, deleted = contrastive_extraction([former, later], 
                                                                        max_ngram=ngram,
                                                                        min_ngram=1, 
                                                                        show_changes=False, 
                                                                        symbols_to_remove=string.punctuation,
                                                                        importance_estimator=sentence_importance.yake_weighted_importance)


    st.write('Keywords:')
    st.table(display_keywords(keywords, top_k))

    st.write('Added Content')
    st.dataframe(added_df(added[0]), use_container_width=True)

    
    html_string1 = highlight_keywords(later, 
                                    keywords[0],
                                    changed_sentences[0],
                                    matched_dicts[0],
                                    added=added[0], 
                                    ngram=1)

    st.write("Monograms Highlighted")
    st.markdown(html_string1, unsafe_allow_html=True)
