import numpy as np
import keras
import h5py



class Text_Generator(keras.utils.Sequence):


	'Generates data for Keras model such that data is evenly distributed '
	def __init__(self, file_names, input_shape = (2500,100), batch_size=32, shuffle=True, training= True):
		'Initialization'

		self.batch_size = batch_size
		self.file_names = file_names
		self.shuffle = shuffle
		self.training = training
		self.indexes = np.arange(len(self.file_names))
		np.random.shuffle(self.indexes)
		self.ind_val = 0

	def __len__(self):
		'Denotes the number of batches per epoch based on the number of augmentations'
		return int(np.floor(len(self.file_names) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		index = self.ind_val 
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [i for i in indexes]

		X, y = self.__data_generation(list_IDs_temp)
		self.ind_val += 1
		self.check_ind_val()
		return X, y

	def check_ind_val(self):

		if self.ind_val == self.__len__():
			np.random.shuffle(self.indexes)
			self.ind_val = 0

	def on_epoch_end(self):
		'Updates indexes after each epoch'

		pass	

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization

		X = np.zeros((self.batch_size,input_shape[0], input_shape[1]))

		for i, ID in enumerate(list_IDs_temp):

			X[i, :, :] = np.array(h5py.File(self.file_names[ID], 'r')['data'])

		Y = np.copy(X)

		return X,Y