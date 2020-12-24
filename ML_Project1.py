#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:04:11 2019

@author: neeleshbhajantri
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#Loading dataset
df = pd.read_csv("wdbc.dataset", header = None)
#taking the dependant coloumn
y = df.iloc[: , 1] 
#replace M,B by 1,0
y = pd.get_dummies(y, drop_first = True)
#removing first 2 coloumns
x_df = df.drop(columns = [0,1])
#Normalisation
x = (x_df - np.min(x_df)) / (np.max(x_df) - np.min(x_df)).values 
#Split train and test
from sklearn.model_selection import train_test_split
#split data into 80-20
xTrain, xRemain, yTrain, yRemain = train_test_split(x, y, test_size = 0.2, random_state = 42)
#Split Test and Validate set
xTest, xVal, yTest, yVal = train_test_split(xRemain, yRemain, test_size = 0.5, random_state = 42)
xTrain = xTrain.T
yTrain = yTrain.T
#Train test
xTest = xTest.T
yTest = yTest.T
xVal = xVal.T
yVal = yVal.T
TP = 0
FP = 0
FN = 0
TN = 0
#Converting to array
aa = yVal.values
bb = yTest.values
#Weights and Bias
def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    #b = np.array([0.0])
    #print(type(b))
    return w
#Sigmoid Func
def sigmoid(z):
    print(z)
    y_head = 1/(1 + np.exp(-z))
    return y_head
 
#Finding z values 
def fwd_prop (w, xTrain, yTrain, y_head):
    v1 = yTrain.T * np.log(y_head)
    v2 = (1-yTrain).T * np.log(1-y_head)
    loss = v1+v2
    cost = (np.sum(-loss)) / xTrain.shape[1]      
    return cost
#Func for derivatives
def back_prop (xTrain,yTrain,y_head):
    #print(xTrain.shape, y_head.shape, yTrain.shape)
    calculated_weight = (np.dot(xTrain,(y_head-yTrain.T)))/xTrain.shape[1]
    #calculated_bias = np.sum(y_head-yTrain)/xTrain.shape[1]
    #gradients = {"calculated_weight": ,"calculated_bias": calculated_bias}
    return calculated_weight
def fwd_back_prop(w,xTrain,yTrain):
    z = np.dot(xTrain.T,w) #+ b
    y_head = sigmoid(z)
    cost = fwd_prop(w,xTrain,yTrain,y_head)
    calculated_weight = back_prop(xTrain,yTrain,y_head)
    return cost,calculated_weight
#Update bias and weights
def learn(w, xTrain, yTrain, alpha, epoch):
    costList = []
    
    for i in range(epoch):
        # execute forward backward propogation to get updated gradients and cost
        cost,calculated_weight = fwd_back_prop(w,xTrain,yTrain)
        costList.append(cost)
        # update weight and bias with the calculated values
        w = w - alpha * calculated_weight
        #b = b - alpha * gradients["calculated_bias"]
    #Updated weights and bias

    #print("Cost lists : " , costList)

    return w, calculated_weight, costList
#Prediction
def predict(w, xVal):
    z = sigmoid(np.dot(w.T,xVal))
    y_prediction = np.zeros((1,xVal.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]> 0.5:
            y_prediction[0,i] = 1
        else:
            y_prediction[0,i] = 0
    return y_prediction
#Accuracy
def logisticRegression(xTrain, yTrain, xVal, alpha ,  epoch):
    # initialize
    size =  xTrain.shape[0]
    w = initialize_weights_and_bias(size)
    parameters, grad, costList = learn(w, xTrain, yTrain, alpha, epoch)
    y_test_predicted = predict(w,xVal)
    #accuracy = 100 - np.mean(np.abs(y_test_predicted - yVal)) * 100
    index = np.arange(epoch)
    #print((costList))
    plt.plot(index, costList)
    #plt.xticks(index,rotation=90) 
    plt.xlabel(" - Iteration Count - ")
    plt.ylabel(" - Accuracy - ")
    plt.show()
    return y_test_predicted
#Func Call
yResul = logisticRegression(xTrain, yTrain, xTest, alpha=0.09, epoch=1)
#print("test accuracy: {} %".format(accuracy))
for j in range(0, yResul.shape[1]):
    if (yResul[0,j]) == 1:
        if (aa[0,j]) == 1:
            TP+= 1
        else:
            FP+=1
    if (yResul[0,j]) == 0:
        if (aa[0,j]) == 0:
            TN+=1
        else:
            FN+=1
            

