# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:09:33 2021

@author: HP
"""

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from Package.layers import *
import numpy as np
from Package.gradient import numerical_gradient
from collections import OrderedDict


class FourLayerNet:

    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3 , output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size_2, hidden_size_3)
        self.params['b3'] = np.zeros(hidden_size_3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size_3, output_size)
        self.params['b4'] = np.zeros(output_size)
        
        # Layer creation
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])

        self.lastLayer = SoftmaxWithLoss()
        
    '''
    def predict(self, x):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)
        
        return y
    '''     

    def predict(self, x):
        for layer in self.layers.values():            
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)    

    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        
        return grads

#net = FourLayerNet(input_size = 2, hidden_size_1=3, hidden_size_2=3, hidden_size_3=3, output_size = 1)

#x = np.random.rand(100,2)
#y = net.predict(x)

#x = np.random.rand(100,2)
#t = np.random.rand(100,1)

#grads = net.numerical_gradient(x,t)

