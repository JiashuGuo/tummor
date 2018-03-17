#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:13:25 2017

@author: jiashu
"""

from numpy.random import seed
randomSeed = 1
seed(randomSeed)
from tensorflow import set_random_seed
set_random_seed(randomSeed)

sample_th = 10
import numpy as np
import random
import matplotlib.pyplot as plt
mos = [1]
ans = [5]
def generate_piecewisecurve():
    from __main__ import freq, mis, num_classes, test_ts
#    freq = 700
#    mis = range(1,101,5) 
#    num_classes=len(range(1,101,5))
    
    window_size = test_ts
    base_path = 'FLAIR_DATA/'
    
    curr_class = 0
    curve_len = np.zeros((num_classes, 1))
    start = True
    plt.figure()
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
                locals()['curve'+str(curr_class)]= np.loadtxt(file_str, delimiter='\t')
                curve_len[curr_class] = locals()['curve'+str(curr_class)].shape[0]   
                mu, sigma = 0, 0.01
                noise = np.random.normal(mu, sigma, locals()['curve'+str(curr_class)].shape[0])
                locals()['curve'+str(curr_class)][:,1] = locals()['curve'+str(curr_class)][:,1] +noise
                #plt.plot(locals()['curve'+str(curr_class)][:,0],locals()['curve'+str(curr_class)][:,1] )
                curr_class += 1          
    
    piece_len = freq*10
    num_piece = int(max(curve_len)/2)//piece_len
       
    shift = 0
    testList = []
    target = []
    while len(testList) < 1:
        label = np.zeros((num_piece, num_classes))
        
        for j in range(num_piece):
        
            if j==0:
                c = [i for i in range(len(curve_len)) if curve_len[i] >= (j+1)*piece_len+ shift]
                #print(c)
                index = random.choice(c)
                selectedCurve = [index]
                rawData = locals()['curve'+str(index)][(j*piece_len + shift):((j+1)*piece_len + shift),:]
                label[j,index]= 1
                #p=rawData((j+1)*piece_len)
        #        plt.vlines(curve0[(j+1)*piece_len+ shift-1,0], 0, 0.8, colors = "k", linestyles = "dashed")
        #        plt.xlabel("Time")
        #        plt.ylabel("Mass")
                
            else:            
                c = [i for i in range(len(curve_len)) if curve_len[i] >= (j+1)*piece_len + shift and i!=selectedCurve[-1]]
                #print(c)
                if len(c) !=0:
                    index = random.choice(c)
                    selectedCurve.append(index)
                    #print(chooseCurve)
                    rawData = np.concatenate((rawData,locals()['curve'+str(index)][(j*piece_len +shift):((j+1)*piece_len+ shift),:]),axis=0)
                    label[j,index]= 1
        #            plt.vlines(curve0[(j+1)*piece_len+ shift-1,0], 0, 0.8, colors = "k", linestyles = "dashed") 
        #            plt.xlabel("Time")
        #            plt.ylabel("Mass")
            
        #plt.plot(rawData[:,0],rawData[:,1],'c')
        #plt.xlabel("Time")
        #plt.ylabel("Mass")
#%%    
    
    pieceData = np.empty (shape=[0,2])
    
    for j in range(len(selectedCurve)-1):
        piece = rawData[j*piece_len:(j+2)*piece_len]       
        num=0
        for k in range(len(piece)):
            if k%freq == 0:              
               pieceData = np.append(pieceData, [[piece[k,0], piece[k,1]]], axis=0)
               num += 1
        if j==0:
            pieceLabel = label[j:(j+2)]
        else:
            pieceLabel = np.concatenate((pieceLabel,label[j:(j+2)]), axis = 0)
    pieceLabel = np.reshape(pieceLabel, (pieceLabel.shape[0]/2, 2, num_classes))
       
    #    plt.figure()       
    #   plt.plot(testData[:,0],testData[:,1])
    #    
    
    te_all = np.reshape(pieceData,(int(pieceData.shape[0]/num),num,pieceData.shape[1]))
    testLabel = np.zeros((te_all.shape[0], 1, num_classes))
    
    for j in range(te_all.shape[0]):
        if j==0:
           testData = te_all[j,0:window_size,1]
        else:
            testData = np.concatenate((testData, te_all[j,0:window_size,1] ), axis=0)
    #    testData = pieceData[j*window_size:]
        
        if window_size <= 10:
            testLabel[j, :,:] = pieceLabel[j, 0, :]
        else:
            testLabel[j, :,:] = pieceLabel[j, 1, :]
    testList.append(testData)
    target.append(testLabel)
    shift = shift + 1
    #%%
Te = [y for x in testList for y in x]
Te=np.array(Te)
Te = np.reshape(Te,(int(Te.shape[0]/window_size), window_size, 1))

Ta = [y for x in target for y in x]
Ta=np.array(Ta)
Ta = np.reshape(Ta,(int(Ta.shape[0]),  num_classes))
    #return Te, Ta
    
