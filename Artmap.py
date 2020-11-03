# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:12:51 2020

@author: Sungbin

name : Fuzzy Artmap
"""

import numpy as np
import os

class Fuzzy_Artmap():
    def __init__(self,M=625,choice=0.001,lr=0.2,vig=0.75):
        self.M = M #feature space
        self.choice=choice # choice parameter
        self.lr=lr # learning rate
        self.vig=vig # vigilance parameter
        self.category=np.array([]) # category 
        self.weight=np.array([]) # T vector
        self.org_label =np.array([]) #original training label for each data
        self.new_cnt =0
        self.temp_list = np.array([]) #remembering the unknown faces
        self.temp_index=0
        self.register_flag=0
    
    #make the complement of the Input I
    def complement_coding(self, I):
        ones = np.ones(I.shape)
        return ones-I
    
    #make the input to I=(I, I^c)
    def make_input(self, I):
        if len(I)==625:
            return np.concatenate((I,self.complement_coding(I)),axis=0)
        return np.concatenate((I,self.complement_coding(I)),axis=1)
    
    # fuzzy min -> elementwise min
    def fuzzy_min(self,a,b):
        if len(a)!=len(b):
            return print("vector length unmatched")
        zip_list = list(zip(a,b))
        
        #find minimum value 
        min_list = [min(i,j) for i,j in zip_list]
        return np.array(min_list)
    
    # To evaluate similarity between input and each category
    #calculate choice function #𝑇_𝑗=|x∧𝐰_𝑗 |+(1−𝛼)(2𝑀−|𝐰_𝑗 |)  (Choice by difference)
    def choice_function(self,x,w):
        T = sum(self.fuzzy_min(x,w))+(1-self.choice)*(2*self.M-sum(w))
        return T
    
    # template matching for fitting
    # To test if the selected category is able to accept the input
    # |𝑅_𝐽⊕𝐈|=𝑀−|𝐱∧𝐰_𝐽 |≤𝑀(1−𝜌)
    def template_matching(self,T_list,I):
        T_max = np.argmax(T_list)
        while 1:
            if sum(T_list)==0:
                #self.category_addition(I)
               
                
                return None, str(self.new_cnt-4)
            T_max = np.argmax(T_list)
            fuz_min = self.fuzzy_min(I,self.weight[T_max])
            resonance = sum(fuz_min)/self.M >=self.vig
            if resonance:
                #self.template_learning(T_max,fuz_min)
                return self.category[T_max], None
            else:
                T_list[T_max]=0
                
    
    def template_learning(self,T_max,new_w):
        self.weight[T_max] = (1-self.lr)*self.weight[T_max]+self.lr*new_w
    
    def category_addition(self,I):
        self.weight = np.concatenate((self.weight,[I]),axis=0)
        self.category = np.concatenate((self.category,[self.new_cnt]))
        
        self.new_cnt+=1
        
        
        
    
    
    def training(self,I):
        #calculate T
           
        for index, x_i in enumerate(I):
            # for the first data, weight is initialize to this input
            if index==0:
                self.weight[0]=x_i
                self.category[0]=self.org_label[0];

                continue
            
            #calculate T
            T_list=[self.choice_function(x_i, w_i) for w_i in self.weight]
            while 1:
                if sum(T_list)==0:
                    self.category = np.concatenate((self.category,[self.org_label[index]]))
                    self.weight = np.concatenate((self.weight,[x_i]),axis=0)
                    break
                
                
                T_max = np.argmax(T_list)
                res = sum(self.fuzzy_min(x_i,self.weight[T_max]))/self.M >=self.vig
                
                if res:
                    if self.org_label[index]==self.category[T_max]:
                        self.weight[T_max]=(1-self.lr)*(self.weight[T_max])+self.lr*(self.fuzzy_min(x_i,self.weight[T_max]))
                        break
                    else:
                        T_list[T_max]=0
                        self.vig = sum(self.fuzzy_min(x_i,self.weight[T_max]))/self.M + 0.01
                else:
                    T_list[T_max]=0
            
            self.vig=0.75
        
    
    def training_register(self,I):
        #calculate T
        self.vig=0.75   
        for index, x_i in enumerate(I):
            
            #calculate T
            T_list=[self.choice_function(x_i, w_i) for w_i in self.weight]
            while 1:
                if sum(T_list)==0:
                    self.category = np.concatenate((self.category,[self.new_cnt]))
                    self.weight = np.concatenate((self.weight,[x_i]),axis=0)
                    break
                
                
                T_max = np.argmax(T_list)
                res = sum(self.fuzzy_min(x_i,self.weight[T_max]))/self.M >=self.vig
                
                if res:
                    if self.new_cnt==self.category[T_max]:
                        self.weight[T_max]=(1-self.lr)*(self.weight[T_max])+self.lr*(self.fuzzy_min(x_i,self.weight[T_max]))
                        break
                    else:
                        T_list[T_max]=0
                        self.vig = sum(self.fuzzy_min(x_i,self.weight[T_max]))/self.M + 0.01
                else:
                    T_list[T_max]=0
            
        
   
        
                
            
        
    def fit(self,train_I,label):
        
        self.M = len(train_I[0])
        self.org_label=label
        #make label category        
        self.category = [0]
        
        #initialize weights to ones
        self.weight = np.ones((1,self.M*2))
        self.temp_list = np.ones((1,self.M*2))

        #make training inputs
        I = self.make_input(train_I)
        
        #training the data
        self.training(I)
        
        #count for new person
        self.new_cnt = self.org_label[-1]+1
        
        print("training done!")
        
    def predict(self, input_i):
        self.vig=0.80
        I =self.make_input(input_i)
        T_list=[self.choice_function(I, w_i) for w_i in self.weight]
        
        return self.template_matching(T_list,I)
    
    def register(self, input_i,name):
        I =self.make_input(input_i)

        if self.register_flag==1:
            self.temp_list[0]=I
            self.register_flag=2
        else:           
            self.temp_list=np.concatenate((self.temp_list,[I]),axis=0)
        
        self.temp_index+=1

        if self.temp_index>30:
            
            self.register_flag=3
            self.temp_index=0
            
            self.training_register(self.temp_list)
            self.new_cnt+=1
            self.temp_list=np.ones((1,self.M*2))
            
            
                 
        







        
        
    
    
        
        
        
