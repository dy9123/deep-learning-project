
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import LSTMmodel
from datetime import datetime
from datetime import timedelta
sns.set()



df = pd.read_csv('RawData.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()



minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))


minmax.transform(df.iloc[:, 1:].astype('float32'))


minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)


num_layers = 5
size_layer = 128
timestamp = 1
epoch = 500
dropout_rate = 0.5
future_day = 50



tf.reset_default_graph()
modelnn = LSTMmodel.Model(0.01, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(df_log.iloc[k: k + timestamp, :].values, axis = 0)
        batch_y = df_log.iloc[k + 1: k + timestamp + 1, :].values
        last_state, _, loss = sess.run([modelnn.last_state, 
                                        modelnn.optimizer, 
                                        modelnn.cost], feed_dict={modelnn.X: batch_x, 
                                                                  modelnn.Y: batch_y, 
                                                                  modelnn.hidden_layer: init_value})
        loss = np.mean(loss)
        init_value = last_state
        total_loss += loss
    total_loss /= (df_log.shape[0] // timestamp)
    if (i + 1) % 2 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)



output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0, :] = df_log.iloc[0, :]
upper_b = (df_log.shape[0] // timestamp) * timestamp
init_value = np.zeros((1, num_layers * 2 * size_layer))
for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[k: k + timestamp, :], axis = 0),
                                     modelnn.hidden_layer: init_value})
    init_value = last_state
    output_predict[k + 1: k + timestamp + 1, :] = out_logits
    
out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[upper_b: , :], axis = 0),
                                     modelnn.hidden_layer: init_value})
init_value = last_state
output_predict[upper_b + 1: df_log.shape[0] + 1, :] = out_logits
df_log.loc[df_log.shape[0]] = out_logits[-1, :]
date_ori.append(date_ori[-1]+timedelta(days=1))



for i in range(future_day - 1):
    out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[-timestamp:, :], axis = 0),
                                     modelnn.hidden_layer: init_value})
    init_value = last_state
    output_predict[df_log.shape[0], :] = out_logits[-1, :]
    df_log.loc[df_log.shape[0]] = out_logits[-1, :]
    date_ori.append(date_ori[-1]+timedelta(days=1))


df_log = minmax.inverse_transform(output_predict)
date_ori=pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()



current_palette = sns.color_palette("Paired", 12)
fig = plt.figure(figsize = (15,10))
ax = plt.subplot(111)
x_range_original = np.arange(df.shape[0])
x_range_future = np.arange(df_log.shape[0])
#ax.plot(x_range_original, df.iloc[:, 1], label = 'true Open', color = current_palette[0])
#ax.plot(x_range_future, df_log[:, 0], label = 'predict Open', color = current_palette[1])
#ax.plot(x_range_original, df.iloc[:, 2], label = 'true High', color = current_palette[2])
#ax.plot(x_range_future, df_log[:, 1], label = 'predict High', color = current_palette[3])
#ax.plot(x_range_original, df.iloc[:, 3], label = 'true Low', color = current_palette[4])#
#ax.plot(x_range_future, df_log[:, 2], label = 'predict Low', color = current_palette[5])
ax.plot(x_range_original, df.iloc[:, 4], label = 'true Close', color = current_palette[6])
ax.plot(x_range_future, df_log[:, 3], label = 'predict Close', color = current_palette[7])
#ax.plot(x_range_original, df.iloc[:, 5], label = 'true Adj Close', color = current_palette[8])
#ax.plot(x_range_future, df_log[:, 4], label = 'predict Adj Close', color = current_palette[9])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc = 'upper center', bbox_to_anchor= (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)
plt.title('overlap stock market')
plt.xticks(x_range_future[::30], date_ori[::30])
plt.show()
plt.savefig('fig1.png')


fig = plt.figure(figsize = (20,8))
plt.subplot(1, 2, 1)
plt.plot(x_range_original, df.iloc[:, 1], label = 'true Open', color = current_palette[0])
plt.plot(x_range_original, df.iloc[:, 2], label = 'true High', color = current_palette[2])
plt.plot(x_range_original, df.iloc[:, 3], label = 'true Low', color = current_palette[4])
plt.plot(x_range_original, df.iloc[:, 4], label = 'true Close', color = current_palette[6])
plt.plot(x_range_original, df.iloc[:, 5], label = 'true Adj Close', color = current_palette[8])
plt.xticks(x_range_original[::60], df.iloc[:, 0].tolist()[::60])
plt.legend()
plt.title('true market')
plt.subplot(1, 2, 2)
plt.plot(x_range_future, df_log[:, 0], label = 'predict Open', color = current_palette[1])
plt.plot(x_range_future, df_log[:, 1], label = 'predict High', color = current_palette[3])
plt.plot(x_range_future, df_log[:, 2], label = 'predict Low', color = current_palette[5])
plt.plot(x_range_future, df_log[:, 3], label = 'predict Close', color = current_palette[7])
plt.plot(x_range_future, df_log[:, 4], label = 'predict Adj Close', color = current_palette[9])
plt.xticks(x_range_future[::60], date_ori[::60])
plt.legend()
plt.title('predict market')
plt.show()
plt.savefig('fig2.png')

import sklearn as sk
from sk.metrics import mean_squared_error
A=df.iloc[:, 4]
B=df_log[:, 3]
mse = mean_squared_error(A, B)

