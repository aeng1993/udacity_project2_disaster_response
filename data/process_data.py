import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' This function load data from the given file pathway and return one dataframe including messages and categories
        Args:
            filepath for the "message" file 
            filepath for the "category" file (describing what categories each message belongs to)
        Outputs:
            df -- a dataframe combing columns from the message and category files         
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    ''' This function take the raw dataframe with message and category columns and perform cleaning steps
        Arg:
            raw dataframe with messages and categories
        Output:
            new dataframe with messages and 36 columns for categories (each column for each category)
            for each category, "1"=message belongs to this category, "0"=message does not belong to this category
    '''
    
    # create a copy of the id for each message, for easy merging later
    df_id = df['id'].copy().to_frame()
    # create a dataframe of the 36 individual category columns
    df_categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = df_categories.iloc[0]
    # use the first row to extract a list of new column names for categories.
    category_colnames = row.str.extract('([A-Za-z_]+)',expand=False)
    # rename the columns of `categories`
    df_categories.columns = category_colnames
    
    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].str.replace('[A-Za-z_]+[-]','', regex=True)
        # convert column from string to numeric
        df_categories[column] = df_categories[column].astype(int)
        
    # add the id column back to the dataframe for easier merging with df dataframe
    df_categories = df_categories.merge(df_id, left_index=True, right_index=True)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # merge the cleaned category columns (one category for each column) to df 
    df = df.merge(df_categories, on='id')
    # remove those rows that have 'related'=2
    # meaning of "2" unclear here. Also, the number of rows is small so okay to remove directly
    df = df[(df['related']==0)|(df['related']==1)]
    #drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df    
        
def save_data(df, database_filepath):
    ''' This function takes the dataframe with cleaned data (df) and a database_filename
        and save the data in df to a database (with database_filename as name)
        Arg:
            df - dataframe with cleaned data
            database_filepath - the filepath of the database, e.g.'DisasterResponse.db'
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('Response', engine, index=False, if_exists='replace') 


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