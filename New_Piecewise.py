

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:25:02 2017

@author: jiashu
"""
from numpy.random import seed
randomSeed=1
seed(randomSeed)
from tensorflow import set_random_seed
set_random_seed(randomSeed)

import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import regularizers
import pandas as pd
import csv  
import argparse
import tensorflow as tf
import time 

import PieceCurve

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-WS", "--windowsize",action='append',nargs='+',type=int,
                    help="the window size for testing")
parser.add_argument("-DS", "--DS_interval",action='append',nargs='+',type=int,
                    help="the downsampling interval for each class")
parser.add_argument("-HD", "--hidden_nodes", action='append', nargs='+',type=int,
                    help="hidden nodes for lstm")
args = parser.parse_args()


windowsize=args.windowsize[0]
print(windowsize)
DS_interval=args.DS_interval[0]
print(DS_interval)
hidden_nodes = args.hidden_nodes[0][0]      #  50, 150, 300, 400
print(hidden_nodes)


 #-----------------------------------------------------------------------------
 #setup some free parameters of the experiment 

data_dim = 1
timesteps = 20         # training timesteps
                       # window_size 
mu, sigma = 0, 0.01
batch_size = 200
lamda = 0
epochs = 40
verbose = 1
threshold=0.0
sample_th=2000
confidence_th=0.95


def perplexity(y_true,y_pred):
    
    cross_entropy = K.mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    perplexity2 = K.pow(2.0, cross_entropy)
    return perplexity2

def numOfcurves(array):
    number = np.ones ((len(array), 1))
    newarray = [sorted(x, reverse=True) for x in array]
    for i in range(len(newarray)):
        sum = newarray[i][0]
        for j in range (1,len(newarray[0])):
            while sum < confidence_th:
               sum += newarray[i][j]
               number[i,0] += 1
    number = np.mean(number)
    return number     
#
# - The first ranges 1 - 5 and denotes the motility rate: number*0.001.
# - The next TWO range from 1 - 10 and denote the angiogenic rate: number*0.05.
# - The last THREE range from 1 - 100 and denote the mitotic rate: number*0.0025

mos = [1]
ans = [5]
mis = range(1,101,5)
#mis=range(1,101,1)

num_classes = 0
for mo in mos:
    for an in ans:
        for mi in mis:
            num_classes += 1

acc_savename=str(num_classes)+'_'+str(randomSeed)+'acc.csv'
perp_savename=str(num_classes)+'_'+str(randomSeed)+'perp.csv'
numcurves_savename=str(num_classes)+'_'+str(randomSeed)+'num.csv'
    
open(acc_savename, 'w').close()
with open(acc_savename, mode='a') as wf1:
    headers = ['ws']+  DS_interval
    writer = csv.writer(wf1)
    writer.writerow(headers)
    
open(perp_savename, 'w').close()
with open(perp_savename, mode='a') as wf2:
    headers = ['ws']+  DS_interval
    writer = csv.writer(wf2)
    writer.writerow(headers)

open(numcurves_savename, 'w').close()
with open(numcurves_savename, mode='a') as wf3:
    headers = ['ws'] + DS_interval
    writer = csv.writer(wf3)
    writer.writerow(headers)


for test_ts in windowsize:
    Acc=[test_ts]
    Perp=[test_ts]
    Num=[test_ts]
    for freq in DS_interval :
        print("\n")
        print("ws:"+str(test_ts)+" interval:"+str(freq)+" hidden_nodes:"+str(hidden_nodes))

  

        
        base_path = 'FLAIR_DATA/'        
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
                        noise = np.random.normal(mu, sigma, Orig_X[:,1].shape)
                        Orig_X [:,1]= Orig_X[:,1]+ noise
                        X = Orig_X
                        new_X = np.empty (shape=[0,2])
                        i= 0
                        while len(new_X) < sample_th and i<len(X):
                            sampling_X = X[i:,:]
                            for j in range(sampling_X.shape[0]-freq):
                                if j%freq == 0:
                                    new_X = np.append(new_X, [[sampling_X[j,0], sampling_X[j,1]]], axis=0)
                            i = i +1
                        #print(len(new_X))
#                        plt.figure()       
#                        plt.plot(X[:,0],X[:,1],label=plot_str)
#                        plt.xlabel("Time")
#                        plt.ylabel("Mass")
#                        plt.legend(loc='center right')
#                        
                        x_all = np.zeros(( np.int(new_X.shape[0]-timesteps+1), timesteps, data_dim))  
                        y_all = np.zeros(( np.int(new_X.shape[0]-timesteps+1), num_classes))  
                        for n in range(new_X.shape[0]-timesteps+1):
                            x_all[n,:,0] = new_X[n:n+timesteps, 1]
                            y_all[n, curr_class] = 1
                        start = False
                        
                    else:
                        # 
                        Orig_X = np.loadtxt(file_str, delimiter='\t')
                        noise = np.random.normal(mu, sigma, Orig_X[:,1].shape)
                        Orig_X [:,1]= Orig_X[:,1]+ noise
                        X = Orig_X
                        new_X = np.empty (shape=[0,2])
                        i= 0
                        while len(new_X) < sample_th and i<len(X):
                            sampling_X = X[i:,:]
                            for j in range(sampling_X.shape[0]-freq):
                                if j%freq == 0:
                                    new_X = np.append(new_X, [[sampling_X[j,0], sampling_X[j,1]]], axis=0)
                            i = i +1
                            
                        #print(len(new_X))
        #                index = np.argsort(new_X[:,0], axis=0)
        #                new_X = new_X[index]
#                        plt.plot(X[:,0],X[:,1],label=plot_str)
#                        plt.xlabel("Time")
#                        plt.ylabel("Mass")
#                        plt.legend(loc='center right')
        
                        x_t = np.zeros(( np.int(new_X.shape[0]-timesteps+1), timesteps, data_dim))  
                        y_t = np.zeros(( np.int(new_X.shape[0]-timesteps+1), num_classes))
                        for n in range(new_X.shape[0]-timesteps+1):
                            x_t[n,:,0] = new_X[n:n+timesteps, 1]
                            y_t[n, curr_class] = 1
        
                        x_all = np.concatenate( (x_all, x_t), axis = 0 )
                        y_all = np.concatenate( (y_all, y_t), axis = 0 )
                        
                    curr_class += 1          
        trainX,testX, trainY, testY = train_test_split(x_all, y_all, test_size=0.2, random_state=1)
        
        #%%
        
        trainX = trainX[(trainX.shape[0]%batch_size):,:,:]
        trainY = trainY[(trainY.shape[0]%batch_size):,:]
        # 3 layers, input size=input_len, output size=1
        
        #kernel_regularizer=regularizers.l2(0.01)
        model = Sequential()
        model.add(LSTM(hidden_nodes, batch_input_shape=(batch_size, timesteps, data_dim), activation='tanh', 
                       recurrent_activation='sigmoid', bias_initializer='Zeros', kernel_initializer='Zeros', 
                       kernel_regularizer=regularizers.l2(lamda),
                       unit_forget_bias=True,stateful=True))
        model.add(Dense(num_classes, activation = 'softmax'))
		
        adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',perplexity])
        
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,verbose=0, shuffle=False)
        # make predictions
        trainPredict = model.predict(trainX, batch_size=batch_size)
       #%%
        
		
        new_model = Sequential()
        new_model.add(LSTM(hidden_nodes, batch_input_shape=(1, None, data_dim), activation='tanh', 
                       recurrent_activation='sigmoid', bias_initializer='Zeros',kernel_initializer='Zeros', 
                       unit_forget_bias=True,stateful=True,kernel_regularizer=regularizers.l2(lamda)))
        new_model.add(Dense(num_classes,  activation = 'softmax'))

        old_weights=model.get_weights()
        new_model.set_weights(old_weights)
        new_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', perplexity])
        new_testX, new_testY= PieceCurve.generate_piecewisecurve()

        loss, accuracy, perp  =new_model.evaluate(new_testX, new_testY, batch_size=1, verbose=0)
        testPredict = new_model.predict(new_testX, batch_size=1)
        a = numOfcurves(testPredict)
           
        print('test accuracy = '+ str(accuracy ) +' test perplexity = '+ str(perp) + ' test num of curves = '+str(a))


        Acc.append(accuracy)
        Perp.append(perp)
        Num.append(a)
    with open(acc_savename, mode='a') as wf1:
        writer = csv.writer(wf1)
        writer.writerow(map(str,Acc))
    with open(perp_savename, mode='a') as wf2:
        writer = csv.writer(wf2)
        writer.writerow(map(str,Perp))
    with open(numcurves_savename, mode='a') as wf3:
        writer = csv.writer(wf3)
        writer.writerow(map(str,Num))


print("execution time--- %s minutes ---" % ((time.time() - start_time)/60))

#acc_figsave=str(num_classes)+'piece_acc'+'.pdf'
#perp_figsave=str(num_classes)+'piece_perp'+'.pdf'
#numcurves_figsave=str(num_classes)+'piece_num'+'.pdf'
#
#
#df1 = pd.DataFrame.from_csv(acc_savename, parse_dates=False)
#plt.figure()
#df1.plot()
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig(acc_figsave,format='pdf')
#plt.show()
#
#df2 = pd.DataFrame.from_csv(perp_savename, parse_dates=False)
#plt.figure()
#df2.plot()
#plt.ylabel('Perplexity')
#plt.legend()
#plt.savefig(perp_figsave,format='pdf')
#plt.show()
#
#df3 = pd.DataFrame.from_csv(numcurves_savename, parse_dates=False)
#plt.figure()
#df3.plot()
#plt.ylabel('Number of curves')
#plt.legend()
#plt.savefig(numcurves_figsave,format='pdf')
#plt.show()


