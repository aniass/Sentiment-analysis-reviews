"""Sentiment analysis model prediction"""

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from joblib import load
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)


MODELS_PATH = 'models\sentiment_model.pkl'


def load_model():
     try:
         with open(MODELS_PATH, 'rb') as file:
             model = load(file)
         return model
     except FileNotFoundError:
         print(f"Error: The model file '{MODELS_PATH}' was not found.")
         return None


def preprocess_text(text):
    # remove punctuation
    words = re.sub("[^a-zA-Z]", " ", text)
    # remove stopwords
    stop_words = stopwords.words('english')
    clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt',
           'jeans', 'pant', 'skirt', 'order', 'white', 'black', 'fabric',
           'blouse', 'sleeve', 'even', 'jacket']
    words = [word.lower() for word in words.split() if word.lower() not in
             stop_words and word.lower() not in clothes]
    # apply lemmatizing
    lem = WordNetLemmatizer()
    words = [lem.lemmatize(word) for word in words]
    return " ".join(words)


def get_prediction(input_text):
    pass
     

if __name__ == '__main__':
    text = input("Type your review of product:\n")
    get_prediction(text)