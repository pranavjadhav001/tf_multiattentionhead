from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf
from attention_head import AttentionHead

class multiAttentionHead(Layer):
    def __init__(self,num_heads=10,k_dim=64,**kwargs):
        self.k_dim = k_dim
        self.num_heads = num_heads
        super(multiAttentionHead,self).__init__(**kwargs)
        
    def build(self,input_shape):
        input_dim = input_shape[-1]
        for i in range(self.num_heads):
            setattr(self,'head_'+str(i),AttentionHead(k_dim=self.k_dim))
            getattr(self,'head_'+str(i),None).build(input_shape)
            
        self.w_o = self.add_weight(name='weight_o',shape=(self.k_dim*self.num_heads,input_dim),\
                                           initializer = 'normal', trainable = True)
        super(multiAttentionHead, self).build(input_shape)
    
    def call(self,input_vec):
        outputs = []
        for i in range(self.num_heads):
            outputs.append(getattr(self,'head_'+str(i),None).call(input_vec))
        final_z = tf.keras.layers.Concatenate(axis=-1)(outputs)
        final_z = K.dot(final_z,self.w_o)
        return final_z      