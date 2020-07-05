# VAE-Job-Analysis

There are 4 files that of interest to the repository currently:
1) custom_losses.py - which houses a custom loss function for the VAE
2) get_model.py - which houses a small VAE model as well as the sampling and VAE loss component
3) generator.py - which houses a data generator that will supply the model with data during training and validation
4) pipeline.py - which houses the main components of the pipeline - converts the indeed .csvs to many .h5 files


There are multiple parameters that are chosen by the user for the pipeline: 
## Parameters

## number of words in dictionary
num_words = 100

## type of embedding used in the text_to_matrix method
mode = 'binary'

## % of data used for validation
validation_size = 0.10

## checking what the max length of words per sample is going to be
max_len = 2500

## batch_size, data used per gradient update
batch_size = 16

## epochs, number of times that the model passes through the data
epochs = 100

## the latent space size - smaller means more compression and less ability to reconstruct
latent_dim = 25
