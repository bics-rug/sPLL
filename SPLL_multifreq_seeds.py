import os
import torch
if __name__ == '__main__':
    for seed in [6]:
        try:
            raise FileNotFoundError
            acc = torch.load(os.path.join(os.getcwd(),'seed_coll',f'acc_coll_{seed}.pt'))
            print('Found seed {}'.format(seed))
        except FileNotFoundError:
            os.system('python3 SPLL_notrain_multifreq.py --epochs=1000 --seed {}'.format(seed,seed))

