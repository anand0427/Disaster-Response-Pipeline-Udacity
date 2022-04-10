import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import sys
import nltk

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    
    '''
    Load and make some small modifications on the data. 
    Input: str - db path
    Output: train and target dataframes, labels
    '''
    sql_engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_table', sql_engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis = 1)
    return X, y, y.columns


def tokenize(text):
    '''
    Tokenizer implementation with some string modifications
    Input: str - text to perform tokenization on
    Output: list - lemmatized strings 
    '''
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    text = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(w) for w in text if not w in stop_words]


def build_model():
    
    '''
    Build vectorizers and transformers and perform gridsearch to get the best parameters with RandomForestClassifier
    Input: None
    Output: GridSearch CV object
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('rf_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'rf_clf__estimator__n_estimators':[25, 30],
    }
    
    cv = GridSearchCV(pipeline, parameters, n_jobs=9, verbose = 4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Test the model to create its classification report
    Input: model object, pd.DataFrame - test df, test target, test labels
    Output: None - prints classification report
    '''
    
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], y_pred[column]))
    
def save_model(model, model_filepath):
    '''
    Dump the model into a pickle file
    Input: model object, str - model filepath
    '''
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)

def main():
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