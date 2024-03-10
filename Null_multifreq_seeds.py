import os
import torch
if __name__ == '__main__':
    for seed in [6]:
        # try:
        #     acc = torch.load(os.path.join(os.getcwd(),'seed_coll',f'acc_coll_{seed}.pt'))
        # except FileNotFoundError:
        os.system('python3 Null_multifreq.py --epochs=1000 --seed {}'.format(seed,seed))

