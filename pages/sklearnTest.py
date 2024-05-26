import random
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from spacy.lang.uk import Ukrainian
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend

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
    text = re.sub(r"""([0-9])([\у0400-\у04FF]|[A-z])""", r"\1 \2", text)
    text = re.sub(r"""([\у0400-\у04FF]|[A-z])([0-9])""", r"\1 \2", text)
    text = re.sub(r"""[\-.,:+*/_]""", ' ', text)
    if lemma:
        return [token.lemma_ for token in nlp(text) if token.is_alpha]
    else:
        return [token.text for token in nlp(text) if token.is_alpha]

st.markdown("# SklearnNlp Page")
st.sidebar.markdown("# SklearnNlp Page")

# Load dataset
@st.cache_data
def load_data():
    land_data = pd.read_csv('land_real_estate.csv').replace(np.nan, "")
    land_data['text'] = land_data['description'] + ' ' + land_data['land_types_source'] + ' ' + land_data['land_types_PcmU']
    return land_data

land_data = load_data()

@st.cache_data
def land_types_download():
    data = pd.read_csv('land_types.csv')
    data = data[['id', 'land_types']]
    data = data.set_index('id')
    data = data.T
    data = data.to_dict('records')
    return data[0]

land_types = land_types_download()

@st.cache_data
def data_augmentation(number):
    unigrams_list = ['продаж', 'земельна', 'ділянка', 'іншої', 'інфраструктура', 'інструменту', 'інвентарю', 'ізумрудє',
                     'івасюка',
                     'яром', 'яворівського', 'шанове', 'чудова', 'цін', 'цільових', 'цілу', 'цілорічно', 'участке',
                     'тільки', 'тухолька',
                     'турківський', 'твердій', 'сухий', 'суха', 'сусідніх', 'сусідні', 'судова', 'сторони', 'сто',
                     'стихії', 'селі', 'села',
                     'сайті', 'руська', 'росташування', 'рокитне', 'розташовану', 'розташований', 'розміщені',
                     'розміщення',
                     'розміщений', 'розмірі', 'розділена', 'покупцю', 'показ', 'повідомлення', 'питання']

    random.seed(0)
    random_words = [unigrams_list[index] for index in random.sample(range(len(unigrams_list)), len(unigrams_list))]
    tsv = 'для ведення товарного сільськогосподарського виробництва'

    # Create an empty list to store the rows
    rows = []
    for sample_index in range(number):
        # Append each new row to the list
        new_row = {'text': tsv + ' ' + " ".join(random.sample(random_words, random.randint(0, 20))), 'land_types': 6,
                   'built_up': 0}
        rows.append(new_row)

    # Concatenate all rows to form the DataFrame
    df = pd.DataFrame(rows)

    # Convert column types
    df = df.astype({'text': 'object', 'land_types': 'int32', 'built_up': 'int32'})

    return df

land_data = pd.concat([land_data, data_augmentation(34)], ignore_index=True)

with st.expander('Show raw data'):
    st.write(land_data)

if st.button('Goooo'):
    X_train, X_test, y_train, y_test = train_test_split(land_data['text'], land_data['built_up'],
                                                        stratify=land_data['built_up'],
                                                        test_size=0.33, random_state=0)

    # Перевірка розподілу категорій перед балансуванням
    st.write("Розподіл категорій перед балансуванням:")
    st.write(y_train.value_counts())

    # Функція для балансування даних
    def balance_data(X, y):
        # Об'єднання даних в один DataFrame
        data = pd.DataFrame({'text': X, 'built_up': y})

        # Розділення даних на класи
        class_0 = data[data['built_up'] == 0]
        class_1 = data[data['built_up'] == 1]

        # Зменшення класу 0 до розміру класу 1
        class_0_downsampled = resample(class_0,
                                       replace=False,
                                       n_samples=len(class_1),
                                       random_state=0)

        # Об'єднання зменшених даних
        balanced_data = pd.concat([class_0_downsampled, class_1])

        X_balanced = balanced_data['text']
        y_balanced = balanced_data['built_up']

        return X_balanced, y_balanced

    # Балансування даних
    X_resampled, y_resampled = balance_data(X_train, y_train)

    # Перевірка розподілу категорій після балансування
    st.write("Розподіл категорій після балансування:")
    st.write(pd.Series(y_resampled).value_counts())

    # Create and fit the pipeline with class weight adjustment
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=ua_tokenizer_lemma)),
        ('svc', LogisticRegression(class_weight={0: 1, 1: 5}))  # Adjust class weight
    ])

    # Використання RandomizedSearchCV для налаштування параметрів
    param_distributions = {'svc__C': [0.1, 1, 10], 'svc__penalty': ['l1', 'l2']}
    with parallel_backend('threading', n_jobs=-1):
        random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(X_resampled, y_resampled)

    st.write(f"Кращі параметри: {random_search.best_params_}")

    y_pred = random_search.predict(X_test)

    # Compute and display the confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=1)
    precision = precision_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write("Confusion Matrix:")
    st.table(cm)
