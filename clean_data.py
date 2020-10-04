# Load libraries
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = stopwords.words('english')
clothes =['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt', 'jeans', 'pant',
          'skirt', 'order', 'white', 'black', 'fabric', 'blouse', 'sleeve', 'even', 'jacket']
lem = WordNetLemmatizer()


def clean_text(words):
    """The function to clean text"""
    words = re.sub("[^a-zA-Z]", " ", words)
    text = words.lower().split()
    return " ".join(text)


def remove_stopwords(review):
    """The function to removing stopwords"""
    text = [word.lower() for word in review.split() if word.lower() not in stop_words and word.lower() not in clothes]
    return " ".join(text)


def remove_numbers(text):
    """The function to removing all numbers"""
    new_text = []
    for word in text.split():
        if not re.search('\d', word):
            new_text.append(word)
    return ' '.join(new_text)


def get_lemmatize(text):
    """The function to apply lemmatizing"""
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)


# Load dataset
url = 'C:\\Python Scripts\\NLP_projekty\\review_final.csv'
dataset = pd.read_csv(url, header=0, index_col=0)

# shape
print(dataset.shape)
print(dataset.head(5))

dataset['Review'] = dataset['Review'].astype(str)
dataset['Review'] = dataset['Review'].apply(clean_text)
dataset['Review'] = dataset['Review'].apply(remove_stopwords)
dataset['Review'] = dataset['Review'].apply(remove_numbers)
dataset['Review'] = dataset['Review'].apply(get_lemmatize)
print(dataset[:5])

data = dataset.to_csv('C:\\Python Scripts\\NLP_projekty\\review_clean.csv', encoding='utf-8')
