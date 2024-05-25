import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from spacy.lang.uk import lemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd
import numpy as np

st.markdown("# SklearnNlp Page")
st.sidebar.markdown("# SklearnNlp Page")

land_data=pd.read_csv('land_real_estate.csv').replace(np.nan, "")
land_data['text']=land_data['description']+' '+land_data['land_types_source']+' '+land_data['land_types_PcmU']

@st.cache_data
def land_types_download():
    data=pd.read_csv('land_types.csv')
    data=data[['id','land_types']]
    data=data.set_index('id')
    data=data.T
    data=data.to_dict('records')
    return (data[0])

land_types=land_types_download()
@st.cache_data
def ua_tokenizer_lemma(text,lemma = True):
    text = re.sub(r"""['’"`�]""", '', text)
    text = re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", r"\1 \2", text)
    text = re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", r"\1 \2", text)
    text = re.sub(r"""[\-.,:+*/_]""", ' ', text)
    if lemma == True:
        return [lemmatizer.UkrainianLemmatizer(word).lemmatize() for word in nltk.word_tokenize(text) if word.isalpha()]
    else:
        return [word for word in nltk.word_tokenize(text) if word.isalpha()]

with st.expander('Show raw data'):
    st.write(land_data)
if st.button('Goooo'):
    X_train_zab, X_test_zab, y_train_zab, y_test_zab = train_test_split(land_data['text'], land_data['built_up'],
                                                                    stratify=land_data['built_up'],
                                                                    test_size=0.33, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(land_data['text'], land_data['land_types'],
                                                    stratify=land_data['land_types'],
                                                    test_size=0.33, random_state=0)

    pipeline = Pipeline([('vectorizer',TfidfVectorizer(tokenizer=ua_tokenizer_lemma)),
                     ('svc',LinearSVC())
                     ])

    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    st.write(accuracy)
    st.write(cm)