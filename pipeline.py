import glob 
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
import keras
import h5py
import os 
import random
import math
import numpy as np 
import generator as gen
from get_model import get_simple_model
from custom_losses import masked_crossentropy

# Parameters
##############
# number of words in dictionary
num_words = 100

# type of embedding used in the text_to_matrix method
mode = 'binary'

# % of data used for validation
validation_size = 0.10

### checking what the max length of words per sample is going to be
max_len = 2500

# batch_size
batch_size = 16

#epochs
epochs = 100

#the latent space size - smaller means more compression and less ability to reconstruct
latent_dim = 25
#############


def pad_array(text, max_len, num_words):
	###
	# array to pad arrays to fit a maximum word length
	###
	len_text = text.shape[0]
	if len_text < max_len:
		padding_length = max_len - len_text

		return np.append(text,-1*np.ones((padding_length, num_words)), axis = 0)
	elif len_text > max_len:
		return text[0:max_len, :]
	else:
		return text 

#Create directory to hold .h5 files for training/analysis
try:
	os.mkdir('../Data')
except:
	print('Data Folder already exists')


### gather all the in the database ###
data_files = glob.glob('../Indeed Job Descriptions/*.csv')

# create the tokenizer
t = Tokenizer(num_words = num_words)

#loop through files to train tokenizer
for file in data_files:

	#read the .csv into a pandas array
	df = pd.read_csv(file, encoding='ISO 8859-1')
	
	#create a list of all the job descriptions
	desc = list(df['Description'].values)

	#train the tokenizer on the descriptions
	t.fit_on_texts(desc)

	# summarize what was learned
	print('Processed....{}'.format(file))
	print('Total # of Descriptions: {}'.format(t.document_count))

	#loop throgh each description in h
	for i in range(len(desc)):

		#pick a filename
		fn = '../Data/' + file.split('/')[-1].split('.')[0] + '_{}.h5'

		#convert the text to a matrix
		out = t.texts_to_matrix(desc[i])

		#padding and normalize the arrays
		out = pad_array(out, max_len, num_words)

		#create a h5 files that contains the matrix representation of the text
		hf = h5py.File(fn, 'w')

		#fill the h5 file with the data
		hf.create_dataset('data', data=out)

		#close the h5 file
		hf.close()

#gather all the files that were created
files = glob.glob('../Data/*.h5')

#shuffle the files 
random.shuffle(files)

#split into training and validation
val_size = math.floor(validation_size*len(files))
training_files = files[val_size:]
validation_files = files[0:val_size]

#create the validation and training generators
training_gen = gen.Text_Generator(training_files, input_shape = (max_len,num_words), batch_size=batch_size)
validation_gen = gen.Text_Generator(validation_files, input_shape = (max_len,num_words), batch_size=batch_size)
		
#use the get_simple_model method to obtain model
AE_model, Encoder_model = get_simple_model(timesteps = max_len, input_dim = num_words, latent_dim = latent_dim)

#create folder for model storage 

try:
	os.makdir('./Models')
except:
	print('Models Directory Already Exists')

#create callbacks that can let us save models midstream
cb = keras.callbacks.ModelCheckpoint('./Models/LSTM_VAE', monitor='val_loss', verbose = 1, save_best_only=True)

#compile the model
model.compile(optimizer='adam', loss=masked_crossentropy)

#fit the model with the generators
history = AE_model.fit_generator(generator=training_gen, validation_data=validation_gen, use_multiprocessing = mp, epochs = epochs,  callbacks=[cb], workers = 8)



		



