# Sentiment analysis of women's clothes reviews 

## General info

The project concerns sentiment analysis of clothes reviews to determine whether the product is recommended or not. Moreover it includes data analysis, data preparation, text mining and build model by using virtue different machine learning and deep learning algorithms. The project also includes EDA analysis and sentiment analysis by using Vader and TextBlob methods.

**I have created the web application in Flask based on this project and code for this app is available [here](https://github.com/aniass/sentiment-app).**

### Dataset
The dataset comes from Woman Clothing Review that can be find at [Kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews). 

## Motivation
The text classification relies to assigned documents into one or many categories. One of the most useful text classification is Sentiment analysis. Is aimed to determine the user's point of view about given product, topic or service. The reviews of products users play a very important role in the e-commerce industry. The product quality can be measured by review given by customer. A new client can decided whether it buy or not given product based on previously reviews.

## Project includes:
* EDA analysis - ***EDA_sentiment.ipynb***
* Sentiment analysis with ML algorithms - ***Sentiment_analysis.ipynb***
* Sentiment analysis with Glove embeddings and LSTM - ***Sentiment_glove.ipynb***
* Sentiment analysis with Vader and TextBlob - ***Sentiment_vader.ipynb***
* Sentiment analysis based on customer rating - ***Sentiment_two.ipynb***
* Python script to clean data - ***clean_data.py*** 
* Python script to use sentiment model with smote method - ***sentiment_model.py***
* data - data used in the project.

## Summary
The goal of the project was sentiment analysis of clothes reviews by using the various methods and determine possible recommendation. I built a model to predict if the review is positive or negative. I have started with the data analysis, data pre-processing and text mining, which cover change text into tokens, remove punctuation, numbers, stop words and normalization them by using lemmatization. I have also used SMOTE method to resolve the imbalance problem in data. In the first approach I used bag of words model to convert the text into numerical feature vectors. To get more accurate predictions I have applied five different classification algorithms like: Logistic Regression, Naive Bayes, Stochastic Gradient Descent, Random Forest and Ada Boosting as well. Finally I  got the best accuracy of 89 % for Logistic Regression model.

In the second I have used a a pre-trained Glove word embeddings with Bidirectional LSTM Neural Network to get more accurate predictions. I achieved an accuracy on the test set equal to 88 % and it is a very good result. From the experiments one can see that the both tested approaches give an overall high accuracy and similar results for the problem.

Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
Logistic Regression | BOW +TF-IDF  | 0.89
Bidirectional LSTM| Glove embeddings  | 0.88
Random Forest| BOW +TF-IDF | 0.87
Naive Bayes | BOW +TF-IDF| 0.87
Stochastic Gradient Descent | BOW +TF-IDF | 0.86
AdaBoosting | BOW +TF-IDF | 0.84

In another approach I also used The Vader and TextBlob methods to analyze the sentiment of reviews and to compare their relevance. 

## Technologies

#### The project is created with:
* Python 3.6
* libraries: pandas, numpy, scikit-learn, keras, tensorflow, nltk, imbalanced-learn, wordcloud, Vader, TextBlob.

#### Running the project:
To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    clean_data.py
    sentiment_model.py

