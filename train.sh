# train parsing generation
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_parsing_gen.py -opt ./configs/parsing_gen.yml

# train parsing tokenization
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_parsing_token.py -opt ./configs/parsing_token.yml

# train vqvae level one
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_vqvae.py -opt ./configs/vqvae_top.yml

# train vqvae level two
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_vqvae.py -opt ./configs/vqvae_bottom.yml

# train index prediction network
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_index_prediction.py -opt ./configs/index_pred_net.yml

# train sampler
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lbie --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python train_sampler.py -opt ./configs/sampler.yml
