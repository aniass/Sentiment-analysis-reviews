# Load libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


URL_DATA  = r'data\review_final.csv'
CLEANED_DATA_PATH = r'data\review_clean.csv'


def read_data(path: str) -> pd.DataFrame:
    """Function to read data"""
    try:
        df = pd.read_csv(path, header=0, index_col=0)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def clean_text(words: str) -> str:
    """Function to clean text"""
    words = re.sub("[^a-zA-Z]", " ", words)
    text = words.lower().split()
    return " ".join(text)


def remove_numbers(text: str) -> str:
    """Function to removing all numbers"""
    new_text = []
    for word in text.split():
        if not re.search('\\d', word):
            new_text.append(word)
    return ' '.join(new_text)


def remove_stopwords(review: str) -> str:
    """Function to removing stopwords"""
    stop_words = stopwords.words('english')
    clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt',
           'jeans', 'pant', 'skirt', 'order', 'white', 'black', 'fabric',
           'blouse', 'sleeve', 'even', 'jacket']
    text = [word.lower() for word in review.split() if word.lower() not in
            stop_words and word.lower() not in clothes]
    return " ".join(text)


def get_lemmatize(text: str) -> str:
    """Function to apply lemmatizing"""
    lem = WordNetLemmatizer()
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)


def preprocess_data(data: str) -> str:
    """Function to preprocess data"""
    data['Review'] = data['Review'].astype(str)
    data['Review'] = data['Review'].apply(clean_text)
    data['Review'] = data['Review'].apply(remove_numbers)
    data['Review'] = data['Review'].apply(remove_stopwords)
    data['Review'] = data['Review'].apply(get_lemmatize)
    return data


if __name__ == '__main__':
    data = read_data(URL_DATA)
    dataset = preprocess_data(data)
    if not dataset.empty:
        print(dataset.shape)
        print(dataset.head(5))
        dataset.to_csv(CLEANED_DATA_PATH, encoding='utf-8')