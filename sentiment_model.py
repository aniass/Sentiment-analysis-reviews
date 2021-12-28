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

stop_words = stopwords.words('english')
clothes =['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt', 'jeans', 'pant',
          'skirt', 'order', 'white', 'black', 'fabric', 'blouse', 'sleeve', 'even', 'jacket']
lem = WordNetLemmatizer()


def text_preprocess(text):
    ''' The function to remove punctuation, 
    stopwords and apply stemming'''
    
    words = re.sub("[^a-zA-Z]"," ", text) 
    
    words = [word.lower() for word in words.split() if word.lower() not in stop_words and word.lower() not in clothes]
    
    words = [lem.lemmatize(word) for word in words]
    return " ".join(words)


# Load dataset
url = 'C:\\Python Scripts\\NLP_projekty\\review_final.csv'
df = pd.read_csv(url, header=0, index_col=0)

# Shape
print(df.shape)
print(df.head())

# Separate into input and output columns
X = df['Review']
y = df['Recommended']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = pd.DataFrame()

classifiers = [
    LogisticRegression(),
    MultinomialNB(),
    SGDClassifier(loss='hinge',penalty='l2', alpha=1e-3, random_state=42),
    RandomForestClassifier(n_estimators=50),
    AdaBoostClassifier(),
    ]

for classifier in classifiers:
    pipe = imbpipeline(steps=[('vect', CountVectorizer(tokenizer=text_preprocess, min_df=5, ngram_range=(1, 2))),
                              ('tfidf', TfidfTransformer()),
                              ('smote', SMOTE()),
                              ('classifier', classifier)])
    
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    
    param_dict = {
                  'Model': classifier.__class__.__name__,
                  'Score': score
                  }
 
    models = models.append(pd.DataFrame(param_dict, index=[0]))

models.reset_index(drop=True, inplace=True)
print(models.sort_values(by='Score', ascending=False))

