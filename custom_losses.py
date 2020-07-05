import keras
import keras.backend as K
import numpy as np 
import tensorflow as tf

def masked_crossentropy(true, pred):
	### custom loss function that will apply a sort of masked cross entropy ###
	true = K.reshape(true, (-1, K.int_shape(true)[-1]))
	pred = K.reshape(pred, (-1, K.int_shape(true)[-1]))

	mask_true = K.clip(K.sum(K.cast(K.not_equal(true, -1 ), K.floatx()), axis = -1, keepdims=False), 0.0, 1.0)
	mask_true = mask_true*K.clip(K.sum(true, axis = -1, keepdims=False), 0.0, 1.0)
	mask_true = K.cast(mask_true, tf.int64)
	inds = K.flatten(tf.where(mask_true >0))

	true = tf.gather(true, inds, axis = 0)
	pred = tf.gather(pred, inds, axis = 0)

	loss = K.categorical_crossentropy(true, pred)

	return K.mean(loss)


if __name__ == '__main__':
	x = np.array([[[0, 1, 0], [0,1,0]], [[0, 1, 0], [-1,-1,-1]]])
	y = np.array([[[0, 1, 0], [0,1,0]], [[0, 1, 0], [0,1,0]]])

	print(x.shape)

	x = K.constant(x)
	y = K.constant(y)

	loss = masked_crossentropy(x,y)
	print(K.eval(loss))

