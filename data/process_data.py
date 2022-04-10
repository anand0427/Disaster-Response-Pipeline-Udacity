import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Takes filepath for messages and categories as *str* and returns a *dataframe* which is a result of a join operation on the dataframes of messages and categories
    '''
    return pd.merge(pd.read_csv(messages_filepath), pd.read_csv(categories_filepath), on='id')


def clean_data(df):
    '''
    splits the column data and creates useful content dataframe.
    
    '''
    categories = df.categories.str.split(';', expand=True)
    category_colnames = [category[:-2] for category in categories.iloc[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.split('-', expand=True)[1]
    
        categories[column] = categories[column].astype('int')
    df = df.drop('categories', axis = 1)
    df = pd.merge(df, categories, left_index=True, right_index=True)
    
    df = df.drop(df[df['related'] == 2].index, axis=0)
    
    
    
    for column in ['id', 'message','original', 'genre']:
        print(column, len(df[column]) - len(df[column].drop_duplicates()))
    df = df.drop_duplicates(subset=['id', 'message','original', 'genre'])
    return df


def save_data(df, database_filename):
    '''
    Write the data in df to database
    '''  
    sql_engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('disaster_table', sql_engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()