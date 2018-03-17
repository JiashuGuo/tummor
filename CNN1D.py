#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:16:42 2018

@author: jiashu
"""
# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import csv  
import os
import argparse
import time
#from utils.utilities import *
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
start_time = time.time()

#parser = argparse.ArgumentParser()
#parser.add_argument("-WS", "--windowsize",action='append',nargs='+',type=int,
#                    help="the window size for testing")
#parser.add_argument("-DS", "--DS_interval",action='append',nargs='+',type=int,
#                    help="the downsampling interval for each class")
#parser.add_argument("-HD", "--hidden_nodes", action='append', nargs='+',type=int,
#                    help="hidden filters for cnn")
#args = parser.parse_args()
#
#
#windowsize=args.windowsize[0]
#print(windowsize)
#DS_interval=args.DS_interval[0]
#print(DS_interval)
#hidden_nodes = args.hidden_nodes[0][0]      #  50, 150, 300, 400
#print(hidden_nodes)

mu, sigma = 0, 0.01
lamda = 0
epochs = 2000
verbose = 1
threshold=0.0
sample_th=1000
confidence_th=0.95
timesteps=20
data_dim=1

batch_size = 600       # Batch size
seq_len = 20          # Number of steps
learning_rate = 0.01
n_channels = 1
filters=30
pool_size=4
strides=1
freq=700
test_ts=20
mos = [1]
ans = [5]

#mis=range(1,101,1)
mis=range(1,101,5)

num_classes = 0
for mo in mos:
    for an in ans:
        for mi in mis:
            num_classes += 1
#            
#acc_savename=str(num_classes)+'cnn'+'acc.csv'
#perp_savename=str(num_classes)+'cnn'+'perp.csv'
#    
#open(acc_savename, 'w').close()
#with open(acc_savename, mode='a') as wf1:
#    headers = ['ws']+  DS_interval
#    writer = csv.writer(wf1)
#    writer.writerow(headers)
#    
#open(perp_savename, 'w').close()
#with open(perp_savename, mode='a') as wf2:
#    headers = ['ws']+  DS_interval
#    writer = csv.writer(wf2)
#    writer.writerow(headers)
#    
#for test_ts in windowsize:
#    #timesteps=test_ts
#    Acc=[test_ts]
#    Perp=[test_ts]
#    
#    for freq in DS_interval :
#        print("\n")
#        print("seqlen:"+str(test_ts)+" interval:"+str(freq)+" filters:"+str(hidden_nodes))

           
    base_path = 'FLAIR_DATA/'        
    curr_class = 0
    start = True
    for mo in mos:
        for an in ans:
            for mi in mis:
    				
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
    
    trainX = trainX[(trainX.shape[0]%batch_size):,:,:]
    trainY = trainY[(trainY.shape[0]%batch_size):,:]
    
    
    
    graph = tf.Graph()
     
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels],
            name = 'inputs')
        labels_ = tf.placeholder(tf.float32, [None, num_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
        
    with graph.as_default():
        # (batch, 20, 1) -> (batch, 2, 5)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=filters, kernel_size=20, strides=1,
        padding='same', activation = tf.nn.relu)
        
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=pool_size, strides=strides, padding='same')
        
        # (batch, 2, 5) -> (batch, 2, 5)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=filters, kernel_size=20, strides=1,
        padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=pool_size, strides=strides, padding='same')
    #    # (batch, 8, 36) -> (batch, kernel_size/strides, filters)
    with graph.as_default():
    # Flatten and add dropout
        flat = tf.reshape(max_pool_2, (-1,(seq_len/strides)*filters))
        print(flat.shape.dims)
        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
        # Predictions
        logits = tf.layers.dense(flat, num_classes)
        print(logits.shape.dims)
        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
        
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        perplexity= tf.exp(cost)
    if (os.path.exists('checkpoints-cnn') == False):
        os.mkdir('checkpoints-cnn')
    train_acc = []
    train_loss = []
    
    with graph.as_default():
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
       
        # Loop over epochs
        for e in range(epochs):
            
            # Loop over batches
            #for x,y in get_batches(X_tr, y_tr, batch_size):
                
                # Feed dictionary
                feed = {inputs_ : trainX, labels_ : trainY, keep_prob_ : 0.8, learning_rate_ : learning_rate}
                
                # Loss
                loss, _ , acc,perp = sess.run([cost, optimizer, accuracy,perplexity], feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)
                
                # Print at each 5 iters
                if (iteration % 5 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc),
                          "Train perp: {:.6f}".format(perp))
                iteration += 1
        saver.save(sess,"checkpoints-cnn/har.ckpt")
    N = timesteps - test_ts +1
    
    new_testX = np.zeros ((N*(testX.shape[0]), test_ts, 1))
    new_testY = np.zeros ((N*(testX.shape[0]), num_classes))
    
    for j in range(testX.shape[0]):
        for i in range(N):
            new_testX[j*N+i,:,0] = testX[j,i:i+test_ts,0]
            new_testY[j*N+i,:]= testY[j, :]
        
    
    new_testX = new_testX[(new_testX.shape[0]%batch_size):,:,:]
    new_testY = new_testY[(new_testY.shape[0]%batch_size):,:]

    test_accuracy = []
    test_perplexity = []
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
        
       
        feed = {inputs_: new_testX,
                labels_: new_testY,
                keep_prob_: 1}
        
        test_acc,test_perp = sess.run([accuracy,perplexity], feed_dict=feed)
        test_accuracy.append(test_acc)
        test_perplexity.append(test_perp)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)),
          "Train perp: {:.6f}".format(np.mean(test_perp)))
    
#    Acc.append(np.mean(test_acc))
#    Perp.append(np.mean(test_perp))
#with open(acc_savename, mode='a') as wf1:
#    writer = csv.writer(wf1)
#    writer.writerow(map(str,Acc))
#with open(perp_savename, mode='a') as wf2:
#    writer = csv.writer(wf2)
#    writer.writerow(map(str,Perp))


print("execution time--- %s minutes ---" % ((time.time() - start_time)/60))

