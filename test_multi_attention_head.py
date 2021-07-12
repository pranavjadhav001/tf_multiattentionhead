import numpy as np
import unittest
import keras.backend as K
from tensorflow.keras.layers import MultiHeadAttention
import tensorflow as tf
from multi_attention_head import multiAttentionHead

class TestmultiAttentionHead(unittest.TestCase):
	def test_output1(self):
		w_q = np.random.rand(5,3,5).astype(np.float32)
		w_k = np.random.rand(5,3,5).astype(np.float32)
		w_v = np.random.rand(5,3,5).astype(np.float32)
		w_o = np.random.rand(3,5,5).astype(np.float32)
		input_vec = np.array([[[0.9,0.2,-0.7,1.6,2.2],\
			[3.2,0.1,-0.5,-1.2,4.0]]]).astype(np.float64)
		input_tensor = tf.keras.Input(shape=[2, 5])
		keras_attention = MultiHeadAttention(num_heads=3,key_dim=5,use_bias=False)
		output_tensor = keras_attention(input_tensor,input_tensor)
		keras_attention.set_weights([w_q,w_k,w_v,w_o])
		keras_attention_output = keras_attention(input_vec,input_vec)

		myAttention = multiAttentionHead(num_heads=3,k_dim=5,use_bias=False)
		output_tensor = myAttention(input_tensor)
		myAttention.set_weights([w_q,w_k,w_v,w_o])
		myAttention_output = myAttention(input_vec)
		self.assertEqual(np.allclose(myAttention_output.numpy(),keras_attention_output.numpy(),0.001,0.001),True)

	def test_output2(self):
		w_q = np.random.rand(5,3,5).astype(np.float32)
		w_k = np.random.rand(5,3,5).astype(np.float32)
		w_v = np.random.rand(5,3,5).astype(np.float32)
		w_o = np.random.rand(3,5,5).astype(np.float32)
		b_q = np.random.rand(3,5).astype(np.float32)
		b_k = np.random.rand(3,5).astype(np.float32)
		b_v = np.random.rand(3,5).astype(np.float32)
		b_o = np.random.rand(5).astype(np.float32)
		input_vec = np.array([[[0.9,0.2,-0.7,1.6,2.2],\
			[3.2,0.1,-0.5,-1.2,4.0]]]).astype(np.float64)
		input_tensor = tf.keras.Input(shape=[2, 5])
		keras_attention = MultiHeadAttention(num_heads=3,key_dim=5,use_bias=True)
		output_tensor = keras_attention(input_tensor,input_tensor)
		keras_attention.set_weights([w_q,b_q,w_k,b_k,w_v,b_v,w_o,b_o])
		keras_attention_output = keras_attention(input_vec,input_vec)

		myAttention = multiAttentionHead(num_heads=3,k_dim=5,use_bias=True)
		output_tensor = myAttention(input_tensor)
		myAttention.set_weights([w_q,b_q,w_k,b_k,w_v,b_v,w_o,b_o])
		myAttention_output = myAttention(input_vec)
		self.assertEqual(np.allclose(myAttention_output.numpy(),keras_attention_output.numpy(),0.001,0.001),True)

	def test_output3(self):
		w_q = np.random.rand(5,3,5).astype(np.float32)
		w_k = np.random.rand(5,3,5).astype(np.float32)
		w_v = np.random.rand(5,3,5).astype(np.float32)
		w_o = np.random.rand(3,5,5).astype(np.float32)
		b_q = np.random.rand(3,5).astype(np.float32)
		b_k = np.random.rand(3,5).astype(np.float32)
		b_v = np.random.rand(3,5).astype(np.float32)
		b_o = np.random.rand(5).astype(np.float32)
		attention_mask = [[[True,False],[True,False]]]
		input_vec = np.array([[[0.9,0.2,-0.7,1.6,2.2],\
			[3.2,0.1,-0.5,-1.2,4.0]]]).astype(np.float64)
		input_tensor = tf.keras.Input(shape=[2, 5])
		keras_attention = MultiHeadAttention(num_heads=3,key_dim=5,use_bias=True)
		output_tensor = keras_attention(input_tensor,input_tensor,)
		keras_attention.set_weights([w_q,b_q,w_k,b_k,w_v,b_v,w_o,b_o])
		keras_attention_output = keras_attention(input_vec,input_vec,attention_mask=tf.cast((attention_mask),tf.int32))

		myAttention = multiAttentionHead(num_heads=3,k_dim=5,use_bias=True)
		output_tensor = myAttention(input_tensor)
		myAttention.set_weights([w_q,b_q,w_k,b_k,w_v,b_v,w_o,b_o])
		myAttention_output = myAttention(input_vec,attention_mask=attention_mask)
		self.assertEqual(np.allclose(myAttention_output.numpy(),keras_attention_output.numpy(),0.001,0.001),True)
		tf.keras.backend.clear_session()

if __name__ == '__main__':
	unittest.main()