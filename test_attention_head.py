from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf
import scipy
import unittest
from attention_head import AttentionHead

class TestAttentionHead(unittest.TestCase):
	def test_output(self):
		input_vec = np.array([[[0.9,0.2,-0.7,1.6,2.2],\
			[3.2,0.1,-0.5,-1.2,4.0]]]).astype(np.float64)
		w_q = np.array([[0.6,0.4],\
		                [-0.4,3.2],\
		                [1.2,3.1],\
		                [2.4,1.8],\
		                [-2.2,0.5]]).astype(np.float64)
		w_k = np.array([[1.8,-0.6],\
		                [0.8,0.8],\
		                [3.3,-0.1],\
		                [-2.4,0.6],\
		                [2.6,0.9]]).astype(np.float64)
		w_v = np.array([[0.6,0.4],\
		                [-0.4,3.2],\
		                [1.2,3.1],\
		                [2.4,1.8],\
		                [-2.2,0.5]]).astype(np.float64)
		q = np.dot(input_vec,w_q)
		k = np.dot(input_vec,w_k)
		v = np.dot(input_vec,w_v)
		score = K.batch_dot(tf.convert_to_tensor(q),tf.convert_to_tensor(k),axes=2)
		score = K.get_value(score)/np.sqrt(2)
		score_soft = scipy.special.softmax(score,axis=2)
		tf_score = K.batch_dot(tf.convert_to_tensor(score_soft),tf.convert_to_tensor(v))
		numpy_score = K.get_value(tf_score)
		dummy = AttentionHead(k_dim=2)
		dummy.build(input_shape=(1,2,5))
		dummy.set_weights([w_q,w_k,w_v])
		keras_score = dummy.call(tf.convert_to_tensor(input_vec,dtype=tf.float32))
		keras_score = K.get_value(keras_score)
		self.assertEqual(np.allclose(numpy_score,keras_score,0.001,0.001),True)

if __name__ == '__main__':
	unittest.main()