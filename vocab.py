from __future__ import division, print_function, absolute_import
import os
import tflearn
from tflearn.data_utils import VocabularyProcessor
import sqlite3
import pandas as pd
import numpy as np

def load_data(database):
    '''
    load data from sql database
    '''
    print('Loading scores and text from {}'.format(database))
    sql_database = sqlite3.connect(database)
    data = pd.read_sql_query("""SELECT Score, Text FROM Reviews""", sql_database)
    score = data['Score']
    text = data['Text']
    return score, text

def make_vocab_processor(name,text,max_length,min_frequency):
    ''''
    generate vocab model
    '''
    print('Making vocabulary model...')
    vp = VocabularyProcessor(max_length, min_frequency=min_frequency)
    vp = vp.fit(text)
    if name == None:
        return vp
    else:
        print('Saving vocabulary model to {}'.format(name))
        vp.save(name)
        return vp

def load_vocab_processor(name,max_length,min_frequency):
    '''
    load model
    '''
    print('Loading vocabulary model from {}'.format(name))
    vp = VocabularyProcessor(max_length, min_frequency=min_frequency)
    vp = vp.restore(name)
    return vp

def get_vocab_processor(text,max_length,min_frequency,name=None):
    '''
    retrieve/make vocab model
    '''
    if name == None:
        vp = make_vocab_processor(name,text,max_length,min_frequency)
        return vp
    else:
        if os.path.isfile(name):
            vp = load_vocab_processor(name,max_length,min_frequency)
            return vp
        else:
            vp = make_vocab_processor(name,text,max_length,min_frequency)
            return vp

def process_vocab(text,vp):
    '''
    apply vocab model to text
    '''
    print('Processing text with vocab model...')
    X = np.array(list(vp.transform(text)))
    return X
