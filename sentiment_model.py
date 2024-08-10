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


def read_data(path: str) -> pd.DataFrame:
    """Read data"""
    try:
        df = pd.read_csv(path, header=0, index_col=0)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()
    

def text_preprocess(text: str) -> str:
    """Remove punctuation, stopwords and apply stemming"""
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


def splitting_data(data: pd.DataFrame):
    """Spliting data into train and test set"""
    data['Review'] = data['Review'].apply(text_preprocess)
    X = data['Review']
    y = data['Recommended']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def calculate_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Calculating models with score"""
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        MultinomialNB(),
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42),
        RandomForestClassifier(n_estimators=50),
        AdaBoostClassifier(),]

    for classifier in classifiers:
        try:
            pipeline = imbpipeline(steps=[
                    ('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('smote', SMOTE()),
                    ('classifier', classifier)
            ])
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            param_dict = {
                      'Model': classifier.__class__.__name__,
                      'Score': score
            }
            models = models.append(pd.DataFrame(param_dict, index=[0]))
        except Exception as e:
            print(f"Error occurred while fitting {classifier.__class__.__name__}: {str(e)}")

    models.reset_index(drop=True, inplace=True)
    models_sorted = models.sort_values(by='Score', ascending=False)
    print(models_sorted)
    return models_sorted


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = splitting_data(df)
    calculate_models(X_train, X_test, y_train, y_test)