"""
Author: Miles Winter
Desc: Perform sentiment analysis on the Amazon Fine Food Reviews
      dataset using a dynamic RNN in TFLearn. 

      Dataset available at:
      https://www.kaggle.com/snap/amazon-fine-food-reviews
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, VocabularyProcessor
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd


#define parameters
max_length = 200    #max length of summary
min_frequency = 2   #min frequency of a word to be included in vp
n_classes = 3       #number of classes
batch_size = 50     #batch size
epochs = 5          #number of training epochs
index = 500000      #number of training samples
n_units = 128       #number of hidden units


#load sql database and select columns from table
print('Loading scores and reviews...')
sql_database = sqlite3.connect('./database.sqlite')
data = pd.read_sql_query("""SELECT Score, Text FROM Reviews""", sql_database)
initial_score = data['Score']
text = data['Text']


#group scores into categories
#1,2 = {0:'bad'}
# 3  = {1:'average'}
#4,5 = {2:'good'}
print('Grouping scores into categories...')
score = np.zeros(len(initial_score))
for i in xrange(len(initial_score)):
    if initial_score[i]<=2:
        score[i]=0.
    elif initial_score[i]==3:
        score[i]=1.
    else:
        score[i]=2.


#generate vocabulary processor and process data
print('Generating vocabulary model...')
vp = VocabularyProcessor(max_length, min_frequency=min_frequency)
X = np.array(list(vp.fit_transform(text)))
n_words = len(vp.vocabulary_) 
Y = to_categorical(score, nb_classes=n_classes)


#split into testing and training sets
trainX, testX, trainY, testY = train_test_split(X, Y,
      train_size=index, random_state=123)


# Define RNN model
net = tflearn.input_data([None, max_length])
net = tflearn.embedding(net, input_dim=n_words, output_dim=n_units)
net = tflearn.gru(net, n_units, dropout=0.8, dynamic=True)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy')


# Train and evaluate on testing set
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          n_epoch=epochs, shuffle=True, batch_size=batch_size)


#save model
model.save('amazon_reviews_rnn.tfl')
