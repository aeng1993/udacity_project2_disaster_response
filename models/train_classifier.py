import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from CountLength import CountLength
import pickle

def load_data(database_filepath):
    ''' 
    INPUT: 
    database filepath, e.g. Emergency_Response.db
    OUTPUT:
    X - messages that need response, as a pandas Series
    Y - categories of the corresponding messages, as a pandas DataFrame (36 categories, 1- positive, 0- negative)
    cat_name - a list of category name for 36 categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Response', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    cat_name = list(Y.columns)
    return X, Y, cat_name

def tokenize(text):
    '''
    This function is a pre-processing step for CountVecterizer. It tokenizes the message.
    
    INPUT:
    text - document (a message that needs response) to be tokenized
    OUTPUT:
    tokens - a list of tokens(words) that has been processed: remove stop words, lemmatize, and convert to lower-case
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    return tokens


def build_model():
    '''
    Build the machine learning model pipeline with GridSearchCV to optimize parameters.
    Tfidf and number of sentences are used as model features.
    OUTPUT:
    The machine learning model pipeline
    '''
    pipeline = Pipeline([
        ('feature', FeatureUnion([
                                    ('text_pipeline', Pipeline([
                                        ('vect', CountVectorizer(max_df=0.7,
                                                                 min_df=15,
                                                                 ngram_range=(1,1),
                                                                 tokenizer=tokenize)),
                                        ('tfidf', TfidfTransformer())])),
                                    ('count_length', CountLength()),
                                ])),
         ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, 
                                                              min_samples_split=8, 
                                                              random_state=1,
                                                              max_features=0.05,
                                                              class_weight='balanced'
                                                             )))
                         ])
    
    parameters = {
                 #'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__max_features': ['auto', 0.05]
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, scoring='f1_weighted')
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluate the model performance on test set, 
    and print the classfication report (precision, recall, f1 score, support)
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save the model as a pickle file to be used for the web app
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    The main function that runs the model pipeline - load data, then build, train, evaluate, and save model.
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
