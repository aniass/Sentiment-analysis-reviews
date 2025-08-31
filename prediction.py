"""Sentiment analysis model prediction"""

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from joblib import load
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)


MODELS_PATH = r'models\sent_model.pkl'


def load_model():
    '''Loading trained sentiment model'''
    try:
        with open(MODELS_PATH, 'rb') as file:
            model = load(file)
        return model
    except FileNotFoundError:
        print(f"Error: The model file '{MODELS_PATH}' was not found.")
        return None


def preprocess_text(text):
    '''Remove punctuation, stopwords and applying lemmatizing on raw data'''
    stop_words = set(stopwords.words('english'))
    clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt',
           'jeans', 'pant', 'skirt', 'order', 'white', 'black', 'fabric',
           'blouse', 'sleeve', 'even', 'jacket']
    words = [word.lower() for word in text if word.lower() not in
             stop_words and word.lower() not in clothes]
    lem = WordNetLemmatizer()
    words = [lem.lemmatize(word) for word in words]
    return words


def get_prediction(input_text):
    '''Generating predictions from raw data'''
    model = load_model()
    if model is None:
        print("Model not loaded. Cannot predict.")
        return
    
    # Preprocess input text
    words = preprocess_text(input_text.split())
    clean_data = [' '.join(words)] # Assuming the model expects a string
    
    # Prediction
    prediction = model.predict([clean_data[0]])
    if prediction == 1:
        answer = "recommended"
    else:
        answer = "not recommended"
    print(f'Your product is {answer}')

     
if __name__ == '__main__':
    text = input("Type your review of product:\n")
    get_prediction(text)
