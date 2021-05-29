import numpy as np
import unittest
import keras.backend as K
import tensorflow as tf
import scipy
from multi_attention_head import multiAttentionHead

class TestmultiAttentionHead(unittest.TestCase):
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
		w_o = np.array([[0.5,-1.2,0.4,0.6,3.2],
		               [2.3,0.2,0.8,0.6,2],
		               [0.5,-1.2,0.4,-1.6,3.2],
		               [0.5,5.2,0.4,4.6,0.2],
		               [0.5,6.2,0.4,0.6,0.5],
		               [0.5,8.2,-0.4,3.0,-0.2]])
		q = np.dot(input_vec,w_q)
		k = np.dot(input_vec,w_k)
		v = np.dot(input_vec,w_v)
		score = K.batch_dot(tf.convert_to_tensor(q),tf.convert_to_tensor(k),axes=2)
		score = K.get_value(score)/np.sqrt(2)
		score_soft = scipy.special.softmax(score,axis=2)
		tf_score = K.batch_dot(tf.convert_to_tensor(score_soft),tf.convert_to_tensor(v))
		numpy_score = K.get_value(tf_score)
		all_numpy_score = np.tile(numpy_score,[1,1,3])
		final_z = np.dot(all_numpy_score,w_o)
		dummy = multiAttentionHead(num_heads=3,k_dim=2)
		dummy.build(input_shape=(1,2,5))
		dummy.set_weights([w_o,w_q,w_k,w_v,\
                 		 w_q,w_k,w_v,\
                  		w_q,w_k,w_v])
		keras_score = dummy.call(tf.convert_to_tensor(input_vec,dtype=tf.float32))
		keras_score = K.get_value(keras_score)
		self.assertEqual(np.allclose(final_z,keras_score,0.001,0.001),True)

if __name__ == '__main__':
	unittest.main()