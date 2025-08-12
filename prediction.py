"""Sentiment analysis model prediction"""

 from nltk.corpus import stopwords
 from nltk.stem import PorterStemmer
 from joblib import load
 import warnings
 warnings.filterwarnings("ignore",category=RuntimeWarning)


 MODELS_PATH = 'models\sentiment_model.pkl'


 def load_model():
     '''Load pretrained model'''
     try:
         with open(MODELS_PATH, 'rb') as file:
             model = load(file)
         return model
     except FileNotFoundError:
         print(f"Error: The model file '{MODELS_PATH}' was not found.")
         return None

        
def get_prediction(input_text):
    pass
     

if __name__ == '__main__':
    text = input("Type your review of product:\n")
    get_prediction(text)