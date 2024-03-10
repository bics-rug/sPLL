import os
import torch
if __name__ == '__main__':
    #seed = 6 
    #os.system('python3 LIF_multifreq.py --epochs=1000 --noise=100 --tqdm_silence --batch_size=1000 --load_optimal --seed {}'.format(seed,seed))

    for seed in [3]:
            os.system('python3 LIF_multifreq.py --layers_size=50 --epochs=500 --load_optimal --seed {}'.format(seed,seed))

                                                                                                                                                                                                                    