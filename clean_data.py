# Load libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = stopwords.words('english')
clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt',
           'jeans', 'pant', 'skirt', 'order', 'white', 'black', 'fabric',
           'blouse', 'sleeve', 'even', 'jacket']
lem = WordNetLemmatizer()


URL_DATA  = 'data\review_final.csv'


def clean_text(words):
    """Function to clean text"""
    words = re.sub("[^a-zA-Z]", " ", words)
    text = words.lower().split()
    return " ".join(text)


def remove_stopwords(review):
    """Function to removing stopwords"""
    text = [word.lower() for word in review.split() if word.lower() not in
            stop_words and word.lower() not in clothes]
    return " ".join(text)


def remove_numbers(text):
    """Function to removing all numbers"""
    new_text = []
    for word in text.split():
        if not re.search('\\d', word):
            new_text.append(word)
    return ' '.join(new_text)


def get_lemmatize(text):
    """Function to apply lemmatizing"""
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)


def read_data(path):
    """Function to read and clean text data """
    data = pd.read_csv(path, header=0, index_col=0)
    data['Review'] = data['Review'].astype(str)
    data['Review'] = data['Review'].apply(clean_text)
    data['Review'] = data['Review'].apply(remove_stopwords)
    data['Review'] = data['Review'].apply(remove_numbers)
    data['Review'] = data['Review'].apply(get_lemmatize)
    return data


if __name__ == '__main__':
    dataset = read_data(URL_DATA)
    print(dataset.shape)
    print(dataset[:5])
    dataset.to_csv('data\review_clean.csv',encoding='utf-8')