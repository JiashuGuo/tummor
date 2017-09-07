
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:50:27 2017

@author: jiashu
"""
import numpy as np

seed = 42
np.random.seed(seed)
#from tensorflow import set_random_seed
#set_random_seed(2)

import keras
import pickle
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
from keras import regularizers
from sklearn.metrics import classification_report
# -----------------------------------------------------------------------------
# setup some free parameters of the experiment 

data_dim = 1
timesteps = 20         # training timesteps
test_ts =  15            # window_size 5, 10 ,15, 20

hidden_nodes = 50      #  50, 150, 300, 400
freq=300                # 150, 300, 700
batch_size = 200
lamda = 0
epochs = 40
verbose = 1
threshold=0.0
sample_th=2000

def perplexity(y_true,y_pred):
    
    cross_entropy = K.mean(K.categorical_crossentropy(y_pred, y_true))
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity
# -----------------------------------------------------------------------------
# 
# 
# 
# - The first ranges 1 - 5 and denotes the motility rate: number*0.001.
# - The next TWO range from 1 - 10 and denote the angiogenic rate: number*0.05.
# - The last THREE range from 1 - 100 and denote the mitotic rate: number*0.0025

mos = [1]
ans = [5]
mis = [1,3,5,10,15,25,35,50,60,75,80,100]


base_path = 'FLAIR_DATA/'
num_classes = 0
for mo in mos:
    for an in ans:
        for mi in mis:
            num_classes += 1

curr_class = 0
start = True
for mo in mos:
    for an in ans:
        for mi in mis:
            #
            file_str = base_path+str(mo)
            if an < 10:
                file_str += '0'
            file_str += str(an)
            if mi < 10:
                file_str += '0'
            if mi < 100:
                file_str += '0'
            plot_str = file_str    
            file_str += str(mi) + '.txt'
            plot_str += str(mi)

            if start:
                Orig_X = np.loadtxt(file_str, delimiter='\t')
                idx=np.where(Orig_X[:,1]>threshold)
                X = Orig_X[idx]
                new_X = np.empty (shape=[0,2])
                i= 0
                while len(new_X) < sample_th and i<len(X):
                    sampling_X = X[i:,:]
                    for j in range(sampling_X.shape[0]-freq):
                        if j%freq == 0:
                            new_X = np.append(new_X, [[sampling_X[j,0], sampling_X[j,1]]], axis=0)
                    i = i +1
                print len(new_X)
#                index = np.argsort(new_X[:,0], axis=0)
#                new_X = new_X[index]
                plt.figure(2)       
                plt.plot(X[:,0],X[:,1],label=plot_str)
                plt.xlabel("Time")
                plt.ylabel("Mass")
                plt.legend(loc='center right')
                
                x_all = np.zeros(( np.int(new_X.shape[0]-timesteps+1), timesteps, data_dim))  
                y_all = np.zeros(( np.int(new_X.shape[0]-timesteps+1), num_classes))  
                for n in range(new_X.shape[0]-timesteps+1):
                    x_all[n,:,0] = new_X[n:n+timesteps, 1]
                    y_all[n, curr_class] = 1
                start = False
                
            else:
                # 
                Orig_X = np.loadtxt(file_str, delimiter='\t')
                idx=np.where(Orig_X[:,1]>threshold)
                X = Orig_X[idx]
                new_X = np.empty (shape=[0,2])
                i= 0
                while len(new_X) < sample_th and i<len(X):
                    sampling_X = X[i:,:]
                    for j in range(sampling_X.shape[0]-freq):
                        if j%freq == 0:
                            new_X = np.append(new_X, [[sampling_X[j,0], sampling_X[j,1]]], axis=0)
                    i = i +1
                    
                print len(new_X)
#                index = np.argsort(new_X[:,0], axis=0)
#                new_X = new_X[index]
                plt.plot(X[:,0],X[:,1],label=plot_str)
                plt.xlabel("Time")
                plt.ylabel("Mass")
                plt.legend(loc='center right')

                x_t = np.zeros(( np.int(new_X.shape[0]-timesteps+1), timesteps, data_dim))  
                y_t = np.zeros(( np.int(new_X.shape[0]-timesteps+1), num_classes))
                for n in range(new_X.shape[0]-timesteps+1):
                    x_t[n,:,0] = new_X[n:n+timesteps, 1]
                    y_t[n, curr_class] = 1

                x_all = np.concatenate( (x_all, x_t), axis = 0 )
                y_all = np.concatenate( (y_all, y_t), axis = 0 )

            # 
            curr_class += 1          
trainX,testX, trainY, testY = train_test_split(x_all, y_all, test_size=0.2, random_state=seed)

#%%

trainX = trainX[(trainX.shape[0]%batch_size):,:,:]
trainY = trainY[(trainY.shape[0]%batch_size):,:]
# 3 layers, input size=input_len, output size=1

#kernel_regularizer=regularizers.l2(0.01)
model = Sequential()
#model.add(Dense(20, batch_input_shape=(batch_size,timesteps,data_dim)))
model.add(LSTM(hidden_nodes, batch_input_shape=(batch_size, timesteps, data_dim), activation='tanh', 
               recurrent_activation='sigmoid', bias_initializer='Zeros', kernel_initializer='Zeros', 
               kernel_regularizer=regularizers.l2(lamda),
               unit_forget_bias=True,stateful=True))
model.add(Dense(num_classes, activation = 'softmax'))
# Train network use squared error loss function
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',perplexity])

model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=False)
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
#acc=mean_absolute_percentage_error(trainY, trainPredict)
#acc = math.sqrt(mean_squared_error(trainY, trainPredict))
#%%
N = timesteps - test_ts +1

new_testX = np.zeros ((N*(testX.shape[0]), test_ts, 1))
new_testY = np.zeros ((N*(testX.shape[0]), num_classes))

for j in range(testX.shape[0]):
    for i in range(N):
        new_testX[j*N+i,:,0] = testX[j,i:i+test_ts,0]
        new_testY[j*N+i,:]= testY[j, :]
    

new_testX = new_testX[(new_testX.shape[0]%batch_size):,:,:]
new_testY = new_testY[(new_testY.shape[0]%batch_size):,:]

# 3 layers, input size=input_len, output size=1
new_model = Sequential()
#new_model.add(Dense(20, batch_input_shape=(batch_size,timesteps,data_dim)))
new_model.add(LSTM(hidden_nodes, batch_input_shape=(batch_size, test_ts, data_dim), activation='tanh', 
               recurrent_activation='sigmoid', bias_initializer='Zeros',kernel_initializer='Zeros', 
               unit_forget_bias=True,stateful=True,kernel_regularizer=regularizers.l2(lamda)))
new_model.add(Dense(num_classes,  activation = 'softmax'))
old_weights=model.get_weights()
new_model.set_weights(old_weights)
new_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', perplexity])
loss, accuracy, perplexity =new_model.evaluate(new_testX, new_testY, batch_size=batch_size, verbose=1)
print 'test accuracy = '+ str(accuracy ) +' test perplexity = '+ str(perplexity) 

testPredict = new_model.predict(new_testX, batch_size=batch_size)
target = np.zeros ((len(new_testY), 1))
prediction = np.zeros ((len(new_testY), 1))
for i in range(len(new_testY)):
    target[i] = np.argmax(new_testY[i,:])+1
    prediction[i] = np.argmax(testPredict[i,:])+1
#    print 'target '+str(target[i])+  '  predition '+str(prediction[i])
target_names = ['Class1', 'Class2','Class3', 'Class4', 'Class5','Class6','7','8','9','10','11','12']        
print(classification_report(target, prediction, target_names=target_names))
#testY  = testY .astype(np.float32)
