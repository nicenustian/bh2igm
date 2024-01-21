# Using Bayesian network to predict Intergatalctic medium gas conditions using supermassive balck holes spectra

This code finds an optimal architecture, search for hyperparameters, trains and make predictions using LYAMAN alpha part of sumilated 1D super massive black hole spectra and converts it to intergalactic medium gas conditions along the line of sight. However, this code easily be utlized for ANY 1D signals using supervised ML. This code utlized deep networks ConvNet, ResNet and MLPNet. 

You need to provide dataset files in a folder and use option --dataset_dir to provide the folder name. The files are written using this snippet. The code calls these fields with the same dictionary. Each data field (such as density)  is given as Number of Examples x Number of samples fashion. For exmaple 5000 x 1024, where 5000 exmaple are provided each with 1024 samples.

# Save multiple named arrays to the same file
```python
data_dict = {'opt': opt, 'density': density, 'temp': temp, 
                 'densityw': densityw, 'tempw': tempw,
                 'weights': weights}

with open(save_file, 'wb') as f:
    np.savez(f, **data_dict)
```


# The output at the end of each layer with 2-layered convolutional network during training
https://github.com/nicenustian/lya-flux-to-density-temp-with-bayesian-networks/assets/111900566/1f08488b-1a3e-46ac-ac50-ae69d0a349d9



# The output at each layer with 2-layered convolutional network during training
https://github.com/nicenustian/lya-flux-to-density-temp-with-bayesian-networks/assets/111900566/416ec95f-07cf-4ee4-811a-4824525dd1da

    
usage: main.py [-h] [--epochs EPOCHS] [--noise NOISE] [--patience_epochs PATIENCE_EPOCHS]
               [--train_fraction TRAIN_FRACTION] [--seed SEED] [--redshift REDSHIFT] [--fwhm FWHM]
               [--hubble HUBBLE] [--omegam OMEGAM] [--skewer_length SKEWER_LENGTH] [--bins BINS]
               [--mean_flux MEAN_FLUX] [--seed_int SEED_INT] [--dataset_dir DATASET_DIR]
               [--output_dir OUTPUT_DIR] [--network NETWORK] [--batch_size BATCH_SIZE] [--lr LR]
               [--features_per_block [FEATURES_PER_BLOCK ...]] [--layers_per_block [LAYERS_PER_BLOCK ...]]

```command

```

Epoch 10 26 [sec]  improve_count = 1
train nll = 0.643188 kll = 0.000000 mae = 0.292537 sigma_cov = 0.978841
test nll = 0.615306 kll = 0.000000 mae = 0.262363 sigma_cov = 0.984722
saving  output/history_densityw.npy
```
