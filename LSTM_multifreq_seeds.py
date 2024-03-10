import os
import torch
if __name__ == '__main__':
    for seed in range(20):
        try:
            acc = torch.load(os.path.join(os.getcwd(),'seed_coll',f'acc_coll_{seed}.pt'))
        except FileNotFoundError:
            os.system('python3 LSTM_multifreq.py --lr_dec=0.01 --lr_spikes=0.01 --epochs=1000 --seed {}'.format(seed,seed))

