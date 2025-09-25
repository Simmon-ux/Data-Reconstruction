#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --time=01:55:00
#SBATCH --mem=10000
###SBATCH -w, --nodelist=g001
module load Python/3.6.6-intel-2018b
source /home/kadow/pytorch/venv/bin/activate
module load CUDA/10.0.130  
export HDF5_USE_FILE_LOCKING='FALSE'

####RAW
#python train.py --root /scratch/kadow/climate/hdf5 --batch_size 18 --n_threads 36 --max_iter 500000 --mask_root /home/kadow/pytorch/pytorch-hdf5-numpy/masks/hadcrut4-missmask.h5 --save_dir ./snapshots/20cr --log_dir ./logs/20cr #--resume /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/200000.pth
####FINE
#python train.py --root /scratch/kadow/climate/hdf5 --mask_root /home/kadow/pytorch/pytorch-hdf5-numpy/masks/hadcrut4-missmask.h5 
--finetune --resume /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/500000.pth --batch_size 18 --n_threads 36 
--max_iter 1000000 --save_dir ./snapshots/20cr --log_dir ./logs/20cr
####TEST
#python test.py --root /scratch/kadow/climate/hdf5/to_test --mask_root /scratch/kadow/climate/hdf5/to_test/mask --snapshot /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/1000000.pth



#python train.py --root F:\py-new-xxzx01\py\AI_sorth\data --batch_size 50 --n_threads 0 --max_iter 6000 --mask_root F:\py-new-xxzx01\py\AI_sorth\masks\train_mask_30000-benshen.h5  --resume F:\py-new-xxzx01\py\AI_sorth\snapshots\xxzx20240306\ckpt\6000.pth --log_dir F:\py-new-xxzx01\py\AI_sorth\logs\xxzx20240306

#python train.py --root D:\py\AI\data --mask_root D:\py\AI\masks\ease_daily-obs-t2m-mask-40N-2011-2020.h5 --finetune --resume D:\py\AI\snapshots\default\ckpt\10000.pth --batch_size 5 --n_threads 0 --max_iter 20000

#python test.py  --root F:\py-new-xxzx01\py\AI_sorth\data --mask_root F:\py-new-xxzx01\py\AI_sorth\masks\ease_test_SH_1979-2023_mask.h5 --snapshot F:\py-new-xxzx01\py\AI_sorth\snapshots\xxzx20240308\ckpt\6000.pth

tensorboard --logdir=xxzx20240517
conda activate python.exe