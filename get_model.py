from keras.models import Model
from keras.layers import *
import keras.backend as K


def get_simple_model(timesteps = 2500, input_dim = 100, latent_dim = 25):
	inputs = Input(shape=(timesteps, input_dim))
	masked = Masking(mask_value=-1)(inputs)
	encoded = LSTM(2*latent_dim)(masked)

	z_mean = Dense(latent_dim)(encoded)
	z_log_sigma = Dense(latent_dim)(encoded)

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

	decoded = Dense(2*latent_dim)(z)
	decoded = RepeatVector(timesteps)(decoded)
	decoded = LSTM(input_dim, activation = 'sigmoid', return_sequences=True)(decoded)

	LSTM_VAE = Model(inputs, decoded)
	vae_loss_ = vae_loss(z_mean, z_log_sigma)
	LSTM_VAE.add_loss(vae_loss_)
	LSTM_VAE.add_metric(vae_loss_, name = 'vae_loss')
	encoder = Model(inputs, z)

	return LSTM_VAE, encoder

def sampling(args):
	z_mean, z_log_sigma = args
	epsilon = K.random_normal(shape=K.shape(z_mean))
	return z_mean + K.exp(z_log_sigma) * epsilon

def vae_loss(z_mean, z_log_sigma, factor = 1e-3):
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return factor*kl_loss


if __name__ == '__main__':
	model,_ = get_simple_model()
	print(model.summary())