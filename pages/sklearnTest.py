import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from spacy.lang.uk import Ukrainian
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd
import numpy as np

# Ensure NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

# Initialize SpaCy's Ukrainian lemmatizer
nlp = Ukrainian()

# Custom tokenizer function with lemmatization
@st.cache_data
def ua_tokenizer_lemma(text, lemma=True):
    text = re.sub(r"""['’"`�]""", '', text)
    text = re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", r"\1 \2", text)
    text = re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", r"\1 \2", text)
    text = re.sub(r"""[\-.,:+*/_]""", ' ', text)
    if lemma:
        return [token.lemma_ for token in nlp(text) if token.is_alpha]
    else:
        return [token.text for token in nlp(text) if token.is_alpha]

st.markdown("# SklearnNlp Page")
st.sidebar.markdown("# SklearnNlp Page")

# Load dataset
land_data = pd.read_csv('land_real_estate.csv').replace(np.nan, "")
land_data['text'] = land_data['description'] + ' ' + land_data['land_types_source'] + ' ' + land_data['land_types_PcmU']

@st.cache_data
def land_types_download():
    data = pd.read_csv('land_types.csv')
    data = data[['id', 'land_types']]
    data = data.set_index('id')
    data = data.T
    data = data.to_dict('records')
    return data[0]

land_types = land_types_download()

with st.expander('Show raw data'):
    st.write(land_data)

if st.button('Goooo'):
    X_train, X_test, y_train, y_test = train_test_split(land_data['text'],land_data['built_up'],
                                                              	stratify=land_data['built_up'],
                                                              	test_size=0.33,random_state=0)

    # Create and fit the pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=ua_tokenizer_lemma)),
        ('svc', LinearSVC())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Compute and display the confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)
