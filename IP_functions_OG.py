# -*- coding: utf-8 -*-
"""
Code from: https://github.com/sul217/MOO_Fairness
author: S. Liu and L.N. Vicente
"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math



# Biobjective formulation: works for any dataset with single binary sensitive attribute, minimize prediction loss and disparate impact, including info of functions and gradients

class Fairness_LogRe_DI_binary:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 2
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        np.random.seed(SEED)
        
        if train_or_test == "train":
            w = np.random.choice(NUM_Alldata, NumData, replace=False)
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "train_ds": # downsample the training data to make two sensitive groups have equal size
            
            g1_idx = np.where(self.Alldata[:, split] == 0)[0]
            g2_idx = np.where(self.Alldata[:, split] == 1)[0]
            w_g1 = np.random.choice(g1_idx, int(NumData*0.5), replace=False)
            w_g2 = np.random.choice(g2_idx, int(NumData*0.5), replace=False)
            w = np.concatenate((w_g1, w_g2))
            
            self.data = self.Alldata[w, :]
            print ("#high-income Female", len(np.where((self.data[:, split] == 0) & (self.data[:, 0] == 1))[0]))
            print ("#high-income Male", len(np.where((self.data[:, split] == 1) & (self.data[:, 0] == 1))[0]))
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test_ds": # downsample the testing data to make two sensitive groups have equal size
            g1_idx = np.where(self.Alldata[:, split] == 0)[0]
            g2_idx = np.where(self.Alldata[:, split] == 1)[0]
            w_g1 = np.random.choice(g1_idx, int(NumData*0.5), replace=False)
            w_g2 = np.random.choice(g2_idx, int(NumData*0.5), replace=False)
            w = np.concatenate((w_g1, w_g2))
            
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            
            print ("#high-income Female", len(np.where((self.data[:, split] == 0) & (self.data[:, 0] == 1))[0]))
            print ("#high-income Male", len(np.where((self.data[:, split] == 1) & (self.data[:, 0] == 1))[0]))
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        elif train_or_test == "test":
            w = np.random.choice(NUM_Alldata, NumData, replace=False)
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("# All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('split', self.split)
        print ('idx', self.idx)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('sum of sensitive', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 
         
    
    ## logistic regression loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize reguarized logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the fairness w.r.t. Disparate impact
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(x,A.T)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each,axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1,sizek)*np.matmul(x, A.T)
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar).reshape(sizek,1)*A
        g23 = (np.sum(g2_each,axis = 1).reshape(1,1)) * (np.sum(g3_each,axis = 0).reshape(1,self.n))
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = np.sum(np.matmul(x, A.T) >= 0)
            sum_zero[i] = count_group[i] - sum_one[i]            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
            
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = 2
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts): 
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        ## get the set of indices according to which the disparate_impact   
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = 2
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        ## Always evaluate each objectives using the whole set of datas
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2
   