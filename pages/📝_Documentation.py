import streamlit as st
from annotated_text import annotated_text
from streamlit_utils import add_lines
from sentence_transformers import SentenceTransformer, util
import pysbd
def compare_semantically(a, b, tau):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    seg = pysbd.Segmenter(language="en", clean=False)

    queries = seg.segment(a)

    # Use the sentences in B as our corpus
    corpus = seg.segment(b)
    if len(queries) > 1 or len(corpus) > 1:
        st.info("Please only include 1 sentences per text area")
    else:
        embeddingsa = model.encode(a)
        embeddingsb = model.encode(b)
        score = round(float(util.dot_score(embeddingsa, embeddingsb)[0][0]), 3)
        if score >= tau:
            st.success(f"sentences match with a score of: {score}")
        else:
            st.error(f"Sentences do not match, since their similarity is too low: {tau} > {score}")

st.markdown("# Documentation")
st.sidebar.markdown('''
# Documentation
* [Sentence Matching](#sentence-matching)
* [Section 2](#section-2)
''', unsafe_allow_html=True)


st.header('Matched Sentences in Context')
st.write('''
This feature provides an overview of the sentence structures of the two versions and shows how they differ.
Put differently it visualizes whether or not sentences present in the earlier version 
are still present in revised version and vice versa. The following Examples show the different relations
sentences from different verisions can have. 
''')

st.markdown("<h4 style='text-align: center;'>Matching Threshold</h4>", unsafe_allow_html=True)
st.write('''
This Hyperparameter can be set by the user in order to give a lower bound on this matching process.
Namely we say two sentences match if their similarity is greater than this lower bound.
The example below shows the effect of chaning the matching threshold.
''')

taua, taub = st.columns(2)

ft = fl = ''
with taua:
    former = st.text_area('Former Version: ', ft, height=100)

with taub:
    Latter = st.text_area('Latter Version: ', fl, height=100)

tau = st.slider("Matching Threshold", 0.0, 1.0, 0.6, 0.01)
compare_semantically(former, Latter, tau)


st.markdown("<h4 style='text-align: center;'>Matched and Changed Sentences</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    changed_a = "Daniels' team has had extensive communications with federal investigators, said a third person familiar with the investigations, who demanded anonymity to discuss the confidential matter. "
    annotated_text((changed_a, f"{25}", "#f2f2f2"))

with col2:
    changed_b = "Daniels' team has had extensive communications with federal investigators, said the third person familiar with the investigation. "
    annotated_text((changed_b, f"{25}", "#f2f2f2"))

st.write("")
st.info("""Non-identical sentences that matched are highlighted in light grey. 
           Furthermore they are indexed by a matching index on the bottom right, this index
           corresponds to the position of the sentence in the latter version.""")

st.markdown("<h4 style='text-align: center;'>New Sentences</h4>", unsafe_allow_html=True)

colnew_a, colnew_b = st.columns(2)

with colnew_a:
    new_former_a = "The payments appear to be part of a pattern of Trump's self-described fixer trying to shield the politician from embarrassing press by buying women's silence. "
    new_former_b = "The new details on the Cohen raid, first reported by The New York Times, emerged as the president boiled over on Twitter about it and evidence that investigators are zeroing in on his inner circle."
    annotated_text((new_former_a, f"{5}", "#f2f2f2"))
    add_lines(18)
    annotated_text((new_former_b, f"{9}", "#f2f2f2"))

with colnew_b:

    new_a = "Investigators also sought bank records and communications with the Trump campaign, according to a third person familiar with the investigation, who also spoke on condition of anonymity to discuss the confidential details. "
    new_b = "The warrants also sought business records on Cohen's dealings in the taxi industry, the person said."
    new_c = "Cohen owns several medallions for New York City yellow cabs that allow them to pick up passengers on the street."
    new_latter_a = "The payments appear to be part of a pattern of Trump's self-described fixer trying to shield the businessman-turned-politician from embarrassing press by buying women's silence. "
    new_latter_b = "The new details on the Cohen raid emerged as the president boiled over on Twitter about it and evidence that investigators are zeroing in on his inner circle."
    annotated_text((new_latter_a, f"{5}", "#f2f2f2"))
    annotated_text((new_a, "NEW", "#00e600"))
    annotated_text((new_b, "NEW", "#00e600"))
    annotated_text((new_c, "NEW", "#00e600"))
    annotated_text((new_latter_b, f"{9}", "#f2f2f2"))
st.write("")
st.info(""" Sentences only present in the latter version are highlighted in green and
        marked using the NEW token.""")



# Removed
st.markdown("<h4 style='text-align: center;'>Removed Sentences</h4>", unsafe_allow_html=True)

colrmv_a, colrmv_b = st.columns(2)

same_a = """The regulations say only Rosenstein, who appointed Mueller last May, has the authority to fire him and only for specific cause.
        Rosenstein has repeatedly said that he has not seen any reason to dismiss Mueller."""

rmv_a = 'White House press secretary Sarah Huckabee Sanders said Tuesday that Trump "certainly believes he has the power" to fire Mueller, though he isn\'t taking that step now.'
rmv_b = 'She echoed Trump\'s frustration, saying he believes federal authorities have "gone too far" by seizing communication between a lawyer and his clients.'
rmv_c = 'The furious president himself blasted out his displeasure early Tuesday, saying on Twitter: "Attorney-client privilege is dead!"'
rmv_d = 'In fact, attorney-client privilege is not absolute and can\'t be invoked when the discussion was part of an effort to commit a crime.'

change_b = 'The search did not appear related to allegations of Russian election interference or possible coordination with the Trump campaign, the main focus of his probe.'
change_a = 'The search did not appear related to allegations of Russian election interference or possible coordination with the Trump campaign, the main focus of Mueller\'s probe.'

with colrmv_a:
    st.write(same_a)
    annotated_text((rmv_a,  "removed", "#ff6666"))
    annotated_text((rmv_b,  "removed", "#ff6666"))
    annotated_text((rmv_c,  "removed", "#ff6666"))
    annotated_text((change_a, f"{3}", "#f2f2f2"))
with colrmv_b:
    st.write(same_a)
    add_lines(14)
    annotated_text((change_b, f"{3}", "#f2f2f2"))
st.write("")
st.info(""" Sentences only present in the former version are highlighted in red and
        marked using the removed token.""")

