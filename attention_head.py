from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf

class AttentionHead(Layer):
    def __init__(self,k_dim=64,**kwargs):
        
        self.k_dim = k_dim
        super(AttentionHead, self).__init__(**kwargs)
    
    def build(self,input_shape):
        input_dim = input_shape[-1]
        self.weight_query = self.add_weight(name='weight_query',shape=(input_dim,self.k_dim),\
                                           initializer = 'normal', trainable = True)
        self.weight_key = self.add_weight(name='weight_key',shape=(input_dim,self.k_dim),\
                                           initializer = 'normal', trainable = True)
        self.weight_value = self.add_weight(name='weight_value',shape=(input_dim,self.k_dim),\
                                           initializer = 'normal', trainable = True)
        super(AttentionHead, self).build(input_shape)
    
    def call(self,input_vec):
        query = K.dot(input_vec,self.weight_query)
        value = K.dot(input_vec,self.weight_value)
        key = K.dot(input_vec,self.weight_key)
        score = K.batch_dot(query,key,axes=2)/K.sqrt(K.cast(self.k_dim,dtype=K.floatx()))
        score = K.softmax(score,axis=2)
        score = K.batch_dot(score,value)
        return score