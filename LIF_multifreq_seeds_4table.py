import os
import torch
if __name__ == '__main__':
    #seed = 6 
    #os.system('python3 LIF_multifreq.py --epochs=1000 --noise=100 --tqdm_silence --batch_size=1000 --load_optimal --seed {}'.format(seed,seed))

    for seed in range(15):
        os.system('python3 LIF_multifreq_finalmodel.py --load_optimal --layers_size=50,50 --seed {}'.format(seed,seed))

                                                                                                                                                                                                                    