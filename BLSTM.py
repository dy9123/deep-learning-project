
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
sns.set()


# In[2]:

class Model:
    def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1):
        
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer)
        backward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple = False)
        forward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple = False)
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop_backward = tf.contrib.rnn.DropoutWrapper(backward_rnn_cells, output_keep_prob = forget_bias)
        forward_backward = tf.contrib.rnn.DropoutWrapper(forward_rnn_cells, output_keep_prob = forget_bias)
        self.backward_hidden_layer = tf.placeholder(tf.float32, shape=(None, num_layers * size_layer))
        self.forward_hidden_layer = tf.placeholder(tf.float32, shape=(None, num_layers * size_layer))
        self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(forward_backward, drop_backward, 
                                                                        self.X, 
                                                                        initial_state_fw = self.forward_hidden_layer, 
                                                                        initial_state_bw = self.backward_hidden_layer,
                                                                        dtype = tf.float32)
        self.outputs = tf.concat(self.outputs, 2)
        rnn_W = tf.Variable(tf.random_normal((size_layer * 2, output_size)))
        rnn_B = tf.Variable(tf.random_normal([output_size]))
        self.logits = tf.matmul(self.outputs[-1], rnn_W) + rnn_B
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


# In[5]:

df = pd.read_csv('update_stresult14-10-14.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()

minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()

num_layers = 1
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.5
future_day = 50


# In[6]:

tf.reset_default_graph()
modelnn = Model(0.01, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    init_value_forward = np.zeros((1, num_layers * size_layer))
    init_value_backward = np.zeros((1, num_layers * size_layer))
    total_loss = 0
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(df_log.iloc[k: k + timestamp, :].values, axis = 0)
        batch_y = df_log.iloc[k + 1: k + timestamp + 1, :].values
        last_state, _, loss = sess.run([modelnn.last_state, 
                                        modelnn.optimizer, 
                                        modelnn.cost], feed_dict={modelnn.X: batch_x, 
                                                                  modelnn.Y: batch_y, 
                                                                  modelnn.backward_hidden_layer: init_value_backward,
                                                                 modelnn.forward_hidden_layer: init_value_forward})
        init_value_forward = last_state[0]
        init_value_backward = last_state[1]
        total_loss += loss
    total_loss /= (df_log.shape[0] // timestamp)
    if (i + 1) % 100 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)


# In[7]:

output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0, :] = df_log.iloc[0, :]
upper_b = (df_log.shape[0] // timestamp) * timestamp
init_value_forward = np.zeros((1, num_layers * size_layer))
init_value_backward = np.zeros((1, num_layers * size_layer))
for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[k: k + timestamp, :], axis = 0),
                                     modelnn.backward_hidden_layer: init_value_backward, modelnn.forward_hidden_layer: init_value_forward})
    init_value_forward = last_state[0]
    init_value_backward = last_state[1]
    output_predict[k + 1: k + timestamp + 1, :] = out_logits
    
out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[upper_b: , :], axis = 0),
                                     modelnn.backward_hidden_layer: init_value_backward, modelnn.forward_hidden_layer: init_value_forward})
init_value_forward = last_state[0]
init_value_backward = last_state[1]
output_predict[upper_b + 1: df_log.shape[0] + 1, :] = out_logits
df_log.loc[df_log.shape[0]] = out_logits[-1, :]
date_ori.append(date_ori[-1]+timedelta(days=1))


# In[8]:

for i in range(future_day - 1):
    out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[-timestamp:, :], axis = 0),
                                     modelnn.backward_hidden_layer: init_value_backward, modelnn.forward_hidden_layer: init_value_forward})
    init_value_forward = last_state[0]
    init_value_backward = last_state[1]
    output_predict[df_log.shape[0], :] = out_logits[-1, :]
    df_log.loc[df_log.shape[0]] = out_logits[-1, :]
    date_ori.append(date_ori[-1]+timedelta(days=1))


# In[9]:

df_log = minmax.inverse_transform(output_predict)
date_ori=pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()

