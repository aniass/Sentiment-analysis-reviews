# Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = 'C:\\Python Scripts\\NLP_projekty\\review_clean.csv'
dataset = pd.read_csv(url, header=0, index_col=0)

# Shape
print(dataset.shape)
print(dataset.head(5))

# Separate into input and output columns
X = dataset['Review']
y = dataset['Recommended']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a numerical feature vector for each document:
vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
X_train_vect = vect.transform(X_train)
print(len(vect.get_feature_names()))

# Create Logistic regression model
model = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression()), ])

model.fit(X_train, y_train)

# Make predictions
ytest = np.array(y_test)
pred_y = model.predict(X_test)

# Evaluate predictions
print('accuracy %s' % accuracy_score(pred_y, y_test))
print(classification_report(ytest, pred_y))