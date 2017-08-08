from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_1d, global_max_pool

def RNN_GRU(max_length,n_words,n_classes,n_units,dynamic=True):
    """define RNN with GRU units"""
    net = tflearn.input_data([None, max_length])
    net = tflearn.embedding(net, input_dim=n_words, output_dim=n_units)
    net = tflearn.gru(net, n_units, dropout=0.8, dynamic=True)
    net = tflearn.fully_connected(net, n_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    return net


def RNN_LSTM(max_length,n_words,n_classes,n_units,dynamic=True):
    """define RNN with LSTM units"""
    net = tflearn.input_data([None, max_length])
    net = tflearn.embedding(net, input_dim=n_words, output_dim=n_units)
    net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
    net = tflearn.fully_connected(net, n_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    return net


def CNN(max_length,n_words,n_classes,n_units):
    '''
    define CNN model
    '''
    net = tflearn.input_data(shape=[None, max_length], name='input')
    net = tflearn.embedding(net, input_dim=n_words, output_dim=n_units)
    branch1 = conv_1d(net, n_units, 3, padding='valid',
                      activation='relu', regularizer="L2")
    branch2 = conv_1d(net, n_units, 4, padding='valid',
                      activation='relu', regularizer="L2")
    branch3 = conv_1d(net, n_units, 5, padding='valid',
                      activation='relu', regularizer="L2")
    net = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
    net = tf.expand_dims(net, 2)
    net = global_max_pool(net)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, n_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    return net


def get_model(name,max_length,n_words,n_classes,n_units,dynamic):
    if name == 'cnn':
        return CNN(max_length,n_words,n_classes,n_units)
    elif name == 'rnn_lstm':
        return RNN_LSTM(max_length,n_words,n_classes,n_units,dynamic=dynamic)
    elif name == 'rnn_gru':
        return RNN_GRU(max_length,n_words,n_classes,n_units,dynamic=dynamic)
    else:
        print("Invalid model: options are 'cnn', 'rnn_lstm', or 'rnn_gru'")
        raise SystemExit
