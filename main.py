"""
Author: Miles Winter
Desc: Perform sentiment analysis on the Amazon Fine Food Reviews
      dataset using a dynamic RNN in TFLearn.

      Dataset available at:
      https://www.kaggle.com/snap/amazon-fine-food-reviews
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
from sklearn.model_selection import train_test_split
import models
import vocab
import argparse

#set up arg parser
p = argparse.ArgumentParser(description="optional arguments")
p.add_argument("-m", "--model", dest="model", type=str, required=True,
               help="Name of model to use: options are 'cnn', 'rnn_gru', or 'rnn_lstm'")
p.add_argument("-l", "--max_length", dest="max_length", type=int, default=200,
               help="max length of text")
p.add_argument("-f", "--frequency", dest="frequency", type=int, default=2,
               help="min frequency of a word to be included in vocab processor")
p.add_argument("-c", "--classes", dest="classes", type=int, default=3,
               help="Number of classes for softmax")
p.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=50,
               help="batch size used for training")
p.add_argument("-e", "--epochs", dest="epochs", type=int, default=5,
               help="Number of training epochs")
p.add_argument("-u", "--units", dest="units", type=int, default=128,
               help="Number of hidden units")
p.add_argument("-s", "--samples", dest="samples", type=int, default=500000,
               help="Number of samples to use for training")
p.add_argument("-v", "--verbosity", dest="verbosity", type=int, default=0,
               help="Tensorboard verbosity value")
p.add_argument("-d", "--dynamic", dest="dynamic", type=bool, default=False,
               help="Whether to use a dynamic rnn")
p.add_argument("-p", "--p_name", dest="p_name", type=str, default=None,
               help="name of vocab processor model")
p.add_argument("-q", "--q_name", dest="q_name", type=str, default='./database.sqlite',
               help="name of vocab processor model")
args = p.parse_args()


#define parameters
max_length = args.max_length
min_frequency = args.frequency
n_classes = args.classes
batch_size = args.batch_size
epochs = args.epochs
index = args.samples
n_units = args.units
vp_model = args.p_name
is_dynamic = args.dynamic
tf_verb = args.verbosity
name = args.model
sql_name = args.q_name


#load data from sql database
initial_score, text = vocab.load_data(sql_name)


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


#convert scores to one hot
Y = to_categorical(score, nb_classes=n_classes)


#get vocabulary processor and process data
vp = vocab.get_vocab_processor(text,max_length,min_frequency,name=vp_model)
X = vocab.process_vocab(text,vp)
n_words = len(vp.vocabulary_)


#split into testing and training sets
trainX, testX, trainY, testY = train_test_split(X, Y,
      train_size=index, random_state=123)


# load selected model
net = models.get_model(name,max_length,n_words,n_classes,n_units,is_dynamic)


# Train and evaluate on testing set
model = tflearn.DNN(net, tensorboard_verbose=tf_verb)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          n_epoch=epochs, shuffle=True, batch_size=batch_size)


#save model
model.save('amazon_{}_model.tfl'.format(name))
