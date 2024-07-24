# Movie - outputs through two layerered ConvNet during training
https://github.com/nicenustian/bh2igm/assets/111900566/9a2ed90b-2d7d-4d88-a74c-9f39815e1a75


# Predict Intergatalctic Medium gas conditions using supermassive black holes spectra

This code finds an optimal architecture, search for hyperparameters (using OPTUNA code), trains and make predictions using LYMAN alpha part of simulated 1D super massive black hole spectra and converts it to intergalactic medium gas conditions along the line of sight. However, this code easily be utilized for ANY 1D signals using supervised ML. This code uses deep networks ConvNet, ResNet and MLPNet. The code can be run on HPC utlizing multiple GPU's on a single node.

# submission script to MPCDF Raven
```command
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out.%j
#SBATCH -e ./out.err.%j

# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J train_gpu
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
# --- uncomment to use 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=500000
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:30:00

### Set the number of CPU cores to match the number of GPUs
##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TF_FORCE_GPU_ALLOW_GROWTH=true

module purge
module load intel/21.3.0
module load anaconda/3/2021.11
module load keras/2.6.0
module load keras-preprocessing/1.1.2
module load cuda/11.4
module load tensorboard/2.8.0
module load tensorflow/gpu-cuda-11.4/2.6.0
module load tensorflow-probability/0.14.1

redshift=4.0
quasars_list=("J120523" "J145147" "J025019" "J021043")
lr=0.008344311976051623
batch_size=1024
l2_factor=0.0008702202857950019
layers_per_block="4 4 4 4 4 4"
features_per_block="8 16 32 32 32 64"

for quasars in ${quasars_list[@]};
do
    for quasar in $quasars;
    do
         echo "$quasar $lr $l2_factor $batch_size $layers_per_block $features_per_block"
                srun python -u main.py --redshift $redshift --l2_factor $l2_factor --output_quantity tempw --quasar $quasar --lr $lr --batch_size $batch_size --layers_per_block $layers_per_block --features_per_block $features_per_block
    done
done
```

You need to provide dataset files in a folder and use option --dataset_dir to provide the folder name. The files are written using the snippet below. The code calls these fields with the same dictionary. Each data field (such as density) is given as Number of examples x Number of samples. Such as 5000 x 1024, where 5000 examples are provided each with 1024 samples. The fields can also be rebinned. The code also calculates 'flux' fields if 'opt' (optical depths are provided). To select fields for params search, training and predictions set --input_quantity and --output_quantity. For this example you can choose temp, density, tempw, densityw, flux and opt. When providing the files in a --dataset_dir you need to specify which files would be make the training dataset which would be only used for predictions. This can can done by adding 'model_train' and 'model_test' as a prefix to dataset file names. Read main.py arguemtns to chnage these prefixes if you need to. 

# Code for writing files
```python
data_dict = {'opt': opt, 'density': density, 'temp': temp, 
                 'densityw': densityw, 'tempw': tempw,
                 'weights': weights}

with open(save_file, 'wb') as f:
    np.savez(f, **data_dict)
```

You can set the related hyperparamters for grid search in OptunaTrainer.py

```python
    def suggest_hyperparams(self):
                
        #suggest a network
        self.network = self.trial.suggest_categorical("network", ["ConvNet", "ResNet"])
        
        #choose hyper parameters for model training
        self.lr = self.trial.suggest_float('lr', 1e-4, 0.5, log=True)
        
        self.l2_factor = self.trial.suggest_float('l2_factor', 1e-4, 1e-1, log=True)

        
        # batch size and features per block in power of two
        self.batch_size = 2**self.trial.suggest_int("batch_size", 10, 13)
        
        self.num_blocks = self.trial.suggest_int("num_blocks", 1, 6)
        
        self.layers_per_block = np.ones(self.num_blocks, dtype=np.int32)
        features_log2 = np.ones(self.num_blocks, dtype=np.int32)
        
        self.layers_per_block[0] = self.trial.suggest_int("layers_per_block1", 1, 4)
        features_log2[0] = self.trial.suggest_int("features_per_block1", 1, 5)
        
        if self.network == "ResNet":
            self.max_layers_per_block = 4
        elif self.network == "ConvNet":
            self.max_layers_per_block = 8
        
    
        for ci in range(1, self.num_blocks):

            self.layers_per_block[ci] = self.trial.suggest_int(
                "layers_per_block"+str(ci+1), 
                self.layers_per_block[ci-1], 
                self.max_layers_per_block
                )
            
            features_log2[ci] = self.trial.suggest_int(
                "features_per_block"+str(ci+1), 
                features_log2[ci-1], 
                features_log2[ci-1]+1
                )
        
        self.features_per_block = np.int32(2**features_log2)
```

```command  
usage: main.py [-h] [--input_quantity INPUT_QUANTITY] [--output_quantity OUTPUT_QUANTITY] [--output_dir OUTPUT_DIR] [--redshift REDSHIFT]
               [--dataset_dir DATASET_DIR] [--dataset_file_filter DATASET_FILE_FILTER] [--prediction_file_filter PREDICTION_FILE_FILTER]
               [--noweights NOWEIGHTS] [--train_fraction TRAIN_FRACTION] [--seed SEED] [--grid_search GRID_SEARCH] [--load_study] [--study_file STUDY_FILE]
               [--trails TRAILS] [--search_epochs SEARCH_EPOCHS] [--search_patience_epochs SEARCH_PATIENCE_EPOCHS] [--epochs EPOCHS]
               [--patience_epochs PATIENCE_EPOCHS] [--load_best_model] [--network NETWORK] [--lr LR] [--batch_size BATCH_SIZE]
               [--layers_per_block [LAYERS_PER_BLOCK ...]] [--features_per_block [FEATURES_PER_BLOCK ...]] [--bins BINS] [--mean_flux MEAN_FLUX]
               [--noise NOISE] [--fwhm FWHM] [--hubble HUBBLE] [--omegam OMEGAM] [--skewer_length SKEWER_LENGTH] [--seed_int SEED_INT]
```

# Predictions using the fields

# using flux field to predict densityw field
![Screenshot 2024-01-21 at 10 35 03](https://github.com/nicenustian/bh2igm/assets/111900566/91dccadd-63bb-4b50-9141-1f2311cc4e5d)


# using flux field to predict tempw field
![Screenshot 2024-01-21 at 10 30 52](https://github.com/nicenustian/bh2igm/assets/111900566/75ec2c55-ff2e-4d4e-9d3f-3b4b2acd128c)

# using density to predict temp field
![Screenshot 2024-01-21 at 10 01 45](https://github.com/nicenustian/bh2igm/assets/111900566/8ed58b74-321b-4e38-a10d-f8cc66fbf961)

# using flux field to predict density field
![Screenshot 2024-01-21 at 10 01 04](https://github.com/nicenustian/bh2igm/assets/111900566/a052a96f-15df-4207-bc09-87b3894e70f4)

# using flux to predict opt field (shown in log scale but tranined in linear scale)
![Screenshot 2024-01-21 at 13 39 05](https://github.com/nicenustian/bh2igm/assets/111900566/e9f30811-b244-4198-84bf-d0d285965253)


## Citation

```
@ARTICLE{2024arXiv240405794N,
       author = {{Nasir}, Fahad and {Gaikwad}, Prakash and {Davies}, Frederick B. and {Bolton}, James S. and {Puchwein}, Ewald and {Bosman}, Sarah E.~I.},
        title = "{Deep Learning the Intergalactic Medium using Lyman-alpha Forest at $ 4 \leq z \leq 5$}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = apr,
          eid = {arXiv:2404.05794},
        pages = {arXiv:2404.05794},
          doi = {10.48550/arXiv.2404.05794},
archivePrefix = {arXiv},
       eprint = {2404.05794},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240405794N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
