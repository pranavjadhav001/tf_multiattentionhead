from tensorflow.python.ops import special_math_ops
import math
import tensorflow as tf
from keras.layers import einsum_dense
#einsumdense use Lambda layer which are non trainable so using base_layer instead of keras.layers.Layer 
from tensorflow.python.keras.engine.base_layer import Layer

class multiAttentionHead(Layer):
    def __init__(self, num_heads=10, k_dim=64, **kwargs):
        self.k_dim = self.q_dim = self.v_dim = k_dim
        self.num_heads = num_heads
        super(multiAttentionHead,self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.f_dim = input_shape[-1]
        #[B,token,feature_dim]*[feature_dim,num_heads,v_dim/q_dim/k_dim]->[B,token,num_heads,q_dim/k_dim/v_dim]
        self.query_dense = einsum_dense.EinsumDense('abc,cde->abde', output_shape=[None, self.num_heads, self.q_dim], bias_axes=None)
        self.key_dense = einsum_dense.EinsumDense('abc,cde->abde', output_shape=[None, self.num_heads, self.k_dim], bias_axes=None)
        self.value_dense = einsum_dense.EinsumDense('abc,cde->abde', output_shape=[None, self.num_heads, self.v_dim], bias_axes=None)
        #[B,token,num_heads,v_dim]*[num_heads,v_dim,feature_dim]->[B,token,feature_dim]
        self.Wo = einsum_dense.EinsumDense('abcd,cde->abe', output_shape=[None, self.f_dim], bias_axes=None)
        super(multiAttentionHead, self).build(input_shape)
    
    def call(self, input_vec):
        query = self.query_dense(input_vec)#[B,token,num_heads,q_dim]
        key = self.key_dense(input_vec)#[B,token,num_heads,k_dim]
        value = self.value_dense(input_vec)#[B,token,num_heads,v_dim]
        #[B,token,num_heads,q_dim]*[B,token,num_heads,k_dim]->[B,num_heads,token,token]
        scaleddotproduct =  special_math_ops.einsum('abcd,aecd->acbe', query, key)
        scaleddotproduct = tf.math.divide(scaleddotproduct, float(math.sqrt(self.k_dim)))
        softmax = tf.nn.softmax(scaleddotproduct, axis=-1)
        #[B,num_heads,token,token]*[B,token,num_heads,v_dim]->[B,token,num_heads,v_dim]
        softmax_value = special_math_ops.einsum('acbe,aecd->abcd', softmax, value)
        #[B,token,num_heads,v_dim]*[num_heads,v_dim,feature_dim]->[B,token,feature_dim]
        final = self.Wo(softmax_value)
        return final