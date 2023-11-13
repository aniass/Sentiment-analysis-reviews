# Load libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


URL_DATA  = r'data\review_final.csv'


def text_preprocess(text):
    ''' Function to remove punctuation,
    stopwords and apply stemming'''
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


def read_data(path):
    """ Function to read and clean text data"""
    data = pd.read_csv(path, header=0, index_col=0)
    return data


def prepare_data(data):
    """ Function to split data on train and test set """
    data['Review'] = data['Review'].apply(text_preprocess)
    X = data['Review']
    y = data['Recommended']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def calculate_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with score '''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        MultinomialNB(),
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42),
        RandomForestClassifier(n_estimators=50),
        AdaBoostClassifier(),]

    for classifier in classifiers:
        pipeline = imbpipeline(steps=[('vect', CountVectorizer(
                                min_df=5, ngram_range=(1, 2))),
                                      ('tfidf', TfidfTransformer()),
                                      ('smote', SMOTE()),
                                      ('classifier', classifier)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        param_dict = {
                      'Model': classifier.__class__.__name__,
                      'Score': score
                     }
        models = models.append(pd.DataFrame(param_dict, index=[0]))

    models.reset_index(drop=True, inplace=True)
    print(models.sort_values(by='Score', ascending=False))


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = prepare_data(df)
    calculate_models(X_train, X_test, y_train, y_test)