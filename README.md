# lya-flux-to-density-temp-with-bayesian-networks

https://github.com/nicenustian/lya-flux-to-density-temp-with-bayesian-networks/assets/111900566/2a04ef9b-0441-4a32-bbf9-fe9873b44ec1


    
usage: main.py [-h] [--epochs EPOCHS] [--noise NOISE] [--patience_epochs PATIENCE_EPOCHS]
               [--train_fraction TRAIN_FRACTION] [--seed SEED] [--redshift REDSHIFT] [--fwhm FWHM]
               [--hubble HUBBLE] [--omegam OMEGAM] [--skewer_length SKEWER_LENGTH] [--bins BINS]
               [--mean_flux MEAN_FLUX] [--seed_int SEED_INT] [--dataset_dir DATASET_DIR]
               [--output_dir OUTPUT_DIR] [--network NETWORK] [--batch_size BATCH_SIZE] [--lr LR]
               [--features_per_block [FEATURES_PER_BLOCK ...]] [--layers_per_block [LAYERS_PER_BLOCK ...]]

```command
(base) mac-n-157:lya-flux-to-density-temp-with-bayesian-networks nasir$ python  main.py --epochs 10
epochs, patience_epochs,  dataset_dir =  10 10 skewers/
Directory 'output__mflux0.4255_fwhm6.00_z4.00/' already exists.

reading/processing file planck1_20_1024_g16_z4.00.npy
<F> = 0.426 0.426 , fdiff = -1.1102230246251565e-16 , bins = 1024

reading/processing file planck1_20_1024_zr750_cold_z4.00.npy
<F> = 0.426 0.426 , fdiff = 4.996003610813204e-16 , bins = 1024

reading/processing file planck1_20_1024_g10_z4.00.npy
<F> = 0.426 0.426 , fdiff = -4.440892098500626e-16 , bins = 1024

reading/processing file planck1_20_1024_zr525_hot_z4.00.npy
<F> = 0.426 0.426 , fdiff = -4.440892098500626e-16 , bins = 1024

reading/processing file planck1_20_1024_z4.00.npy
<F> = 0.426 0.426 , fdiff = -2.220446049250313e-16 , bins = 1024

reading/processing file planck1_20_1024_zr750_hot_z4.00.npy
<F> = 0.426 0.426 , fdiff = -1.6653345369377348e-16 , bins = 1024

reading/processing file planck1_20_1024_zr675_hot_z4.00.npy
<F> = 0.426 0.426 , fdiff = -5.551115123125783e-17 , bins = 1024

reading/processing file planck1_20_1024_g14_z4.00.npy
<F> = 0.426 0.426 , fdiff = -5.551115123125783e-16 , bins = 1024

reading/processing file planck1_20_1024_zr525_z4.00.npy
<F> = 0.426 0.426 , fdiff = -5.551115123125783e-17 , bins = 1024

reading/processing file planck1_20_1024_zr675_cold_z4.00.npy
<F> = 0.426 0.426 , fdiff = 4.440892098500626e-16 , bins = 1024

reading/processing file planck1_20_1024_zr525_cold_z4.00.npy
<F> = 0.426 0.426 , fdiff = -2.220446049250313e-16 , bins = 1024

reading/processing file planck1_20_1024_zr750_z4.00.npy
<F> = 0.426 0.426 , fdiff = -2.220446049250313e-16 , bins = 1024

reading/processing file planck1_20_1024_zr675_z4.00.npy
<F> = 0.426 0.426 , fdiff = -1.1102230246251565e-16 , bins = 1024

before scaling mean (flux, densityw, tempw, weights)  0.4255000054836276 -0.14893177990218118 4.010769650750928 1.0
scalers= [0.42550001] [0.10649995] [-0.14893178] [0.25638148] [4.01076965] [0.04357046]
after scaling mean (flux, densityw, tempw, weights) -2.482389080618383e-15 2.815457269032497e-16 -1.5269488237925963e-14
datasets shapes  (65000, 1024) (65000, 1024) (65000, 1024) (65000, 1024)
Metal device set to: Apple M1 Pro

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

Random seed set as 12345
noisey noise 0.02
fixed snr= 50.00000111758712
network, layers, features, units, lr, batch_size =  ConvNet [2 2] [32 64] 1024 1e-04 256
saving weights.. improved from 1508.3457 to 1508.3457 output/nnweights_densityw/

Epoch 1 29 [sec]  improve_count = 0
train nll = 1.075413 kll = 0.000000 mae = 0.550698 sigma_cov = 0.824938
test nll = 1.472994 kll = 0.000000 mae = 0.638882 sigma_cov = 0.577753
saving weights.. improved from 871.1227 to 871.1227 output/nnweights_densityw/

Epoch 2 27 [sec]  improve_count = 0
train nll = 0.864903 kll = 0.000000 mae = 0.450245 sigma_cov = 0.952655
test nll = 0.850706 kll = 0.000000 mae = 0.439685 sigma_cov = 0.866324
saving weights.. improved from 829.62756 to 829.62756 output/nnweights_densityw/

Epoch 3 26 [sec]  improve_count = 0
train nll = 0.828322 kll = 0.000000 mae = 0.427417 sigma_cov = 0.962731
test nll = 0.810183 kll = 0.000000 mae = 0.405595 sigma_cov = 0.967686

Epoch 4 25 [sec]  improve_count = 1
train nll = 0.799191 kll = 0.000000 mae = 0.407971 sigma_cov = 0.966051
test nll = 0.819911 kll = 0.000000 mae = 0.412918 sigma_cov = 0.974807
saving weights.. improved from 797.83124 to 797.83124 output/nnweights_densityw/

Epoch 5 26 [sec]  improve_count = 0
train nll = 0.773802 kll = 0.000000 mae = 0.389170 sigma_cov = 0.968236
test nll = 0.779132 kll = 0.000000 mae = 0.386440 sigma_cov = 0.977674
saving weights.. improved from 778.4815 to 778.4815 output/nnweights_densityw/

Epoch 6 27 [sec]  improve_count = 0
train nll = 0.747497 kll = 0.000000 mae = 0.370237 sigma_cov = 0.970261
test nll = 0.760236 kll = 0.000000 mae = 0.370071 sigma_cov = 0.980708

Epoch 7 26 [sec]  improve_count = 1
train nll = 0.722184 kll = 0.000000 mae = 0.351067 sigma_cov = 0.972644
test nll = 0.760410 kll = 0.000000 mae = 0.390673 sigma_cov = 0.933137
saving weights.. improved from 651.4989 to 651.4989 output/nnweights_densityw/

Epoch 8 27 [sec]  improve_count = 0
train nll = 0.696751 kll = 0.000000 mae = 0.331601 sigma_cov = 0.974868
test nll = 0.636229 kll = 0.000000 mae = 0.291422 sigma_cov = 0.979366
saving weights.. improved from 627.0443 to 627.0443 output/nnweights_densityw/

Epoch 9 27 [sec]  improve_count = 0
train nll = 0.670220 kll = 0.000000 mae = 0.312060 sigma_cov = 0.977134
test nll = 0.612348 kll = 0.000000 mae = 0.275612 sigma_cov = 0.971390

Epoch 10 26 [sec]  improve_count = 1
train nll = 0.643188 kll = 0.000000 mae = 0.292537 sigma_cov = 0.978841
test nll = 0.615306 kll = 0.000000 mae = 0.262363 sigma_cov = 0.984722
saving  output/history_densityw.npy
```
