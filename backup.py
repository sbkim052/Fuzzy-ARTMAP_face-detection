# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:12:51 2020

@author: Sungbin

name : Fuzzy Artmap
"""

import numpy as np

class Fuzzy_Artmap():
    def __init__(self,M=128,choice=0.00001,lr=0.2,vig=0.5):
        self.M = M #feature space
        self.choice=choice # choice parameter
        self.lr=lr # learning rate
        self.vig=vig # vigilance parameter
        self.category=np.array([]) # category 
        self.weight=np.array([[]]) # T vector
        self.org_label =np.array([]) #original training label for each data
    
    #make the complement of the Input I
    def complement_coding(self, I):
        ones = np.ones(I.shape)
        return ones-I
    
    #make the input to I=(I, I^c)
    def make_input(self, I):
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
    
    #select the most similar category
    def code_competition(self, T_list):
        f = lambda i: T_list[i]
        J=max(range(len(output)), key=f)
        
        return J
        """
        for x_i in x:
            w 리스트에 있는 값과 함께 T리스트를 전부 계산한다
            T리스트 중 가장 큰 값을 찾는다.
            T 리스트의 값과 resonace 계산을 한다.
            만약 resonace에 부합하지 안흔다면 다시 T리스트 계산 값중 큰 값을 찾는다.
        """
    # template matching for fitting
    # To test if the selected category is able to accept the input
    # |𝑅_𝐽⊕𝐈|=𝑀−|𝐱∧𝐰_𝐽 |≤𝑀(1−𝜌)
    def template_matching_f(self,x,w):
        T_list=[self.choice_function(x, w_i) for w_i in self.w_list]
        J=self.code_competition(T_list)
    
    # template matching for testing    
    def template_matching_t(self,x,w):
        print("jello")
      
    def training(self,I):
        #calculate T
        cnt =0 # count for category initialization
        
        
        for index, x_i in enumerate(I):
            print(index)
            flag=0 #flag for non classified
            # for the first data, weight is initialize to this input
            if index==0:
                flag=1
                self.weight=[x_i]
                self.category=np.append(self.category,self.org_label[index])
                cnt+=1
            else:
                T_list=[self.choice_function(x_i, w_i) for w_i in self.weight]
            
                for i in range(cnt):
                    
                    T_max = np.argmax(T_list)
                    resonance = self.M-sum(self.fuzzy_min(x_i,self.weight[T_max]))<=self.M*(1-self.vig)

                    if resonance:

                        if self.org_label[index]==self.category[T_max]:
                            print("resonance")
                            flag=1
                            self.weight[T_max]=(1-self.lr)*(self.weight[T_max])+self.lr*(self.fuzzy_min(x_i,self.weight[T_max]))
                            break
                        else:
                            print("no resonance")
                            T_list[T_max]=0
                            self.vig = sum(self.fuzzy_min(x_i,self.weight[T_max]))/self.M + 0.01
                    else:
                        continue
            if flag==0:
                print("weight plus")
                self.weight=np.append(self.weight,[x_i],axis=0)
                self.category=np.append(self.category,self.org_label[index])
                cnt+=1
                        
                    
            #print(index, " ",self.weight)
                    
            
            
            #else:
                #check the resonance condition
                #boolean = self.M-sum(fuzzy_min())
        
    def fit(self,train_I,label):
        self.M = len(train_I[0])
        self.org_label=label

        #make training inputs
        I = self.make_input(train_I)
        
        #training the data
        self.training(I)
        
        print("training done!")
        
    def predict(self, input_i):
        I =self.make_input(input_i)







        
        
    
    
        
        
        