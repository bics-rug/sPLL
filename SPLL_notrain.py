import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
# from bokeh.plotting import figure, show, row, output_notebook
# from bokeh.plotting import output_notebook
# output_notebook()
import tqdm
# import seaborn as sns
from matplotlib.colors import ListedColormap
import itertools
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
import nni
import logging
import os
from sklearn.metrics import multilabel_confusion_matrix
import json
from utils import *
from models import *

LOG = logging.getLogger('snn')
import re
from sklearn.metrics import accuracy_score
def main(args):
    parameters = vars(args)  # copy by reference (checked below)
    if args.load_optimal:
        opt_params = json.load(open('SPLL_multifreq_opt.txt', 'r'))
        print('Loading opt params', opt_params)
        for key, val in opt_params.items():
            parameters[key] = val
            LOG.debug(print(key))
            assert (args.__dict__[key] == parameters[key])
    parameters['n_in'] = 1
    print(parameters)
    np.random.seed(args.seed)
    freqs1 = np.arange(30, 100, 9)
    freqs2 = np.arange(30, 100, 27)
    # parameters = {}
    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    print('device in use:', parameters['device'])

    trials_desc = str(args.trials_n)
    noise_desc = str(args.noise)
    shift_desc = str(args.shift)
    sprinkle_desc = str(args.sprinkle)
    seed_desc = str(args.seed)
    sim_time_desc = str(parameters['sim_time'])
    try:
        os.mkdir('synthetic_data')
    except FileExistsError:
        pass

    freq1_desc = 'f1_' + str(freqs1[0]) + '_' + str(freqs1[1] - freqs1[0]) + '_' + str(freqs1[-1])
    if freqs2 is not None:
        freq2_desc = 'f2_' + str(freqs2[0]) + '_' + str(freqs2[1] - freqs2[0]) + '_' + str(freqs2[-1])
    else:
        freq2_desc = ''
    print('freqs2',freqs2)
    trials_desc = str(args.trials_n)
    noise_desc = str(args.noise)
    shift_desc = str(args.shift)
    sprinkle_desc = str(args.sprinkle)
    seed_desc = str(args.seed)
    sim_time_desc = str(parameters['sim_time'])
    try:
        os.mkdir('synthetic_data')
    except FileExistsError:
        pass
    file_spikes = 'in_spikes_m-' + freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + '-' + sprinkle_desc + '_' + sim_time_desc + '-' + seed_desc + '.pt'
    file_labels_combined = 'labels_combined-' + freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + sprinkle_desc + '-' + sim_time_desc + '-' + seed_desc + '.pt'
    print('noise: ' + str(args.noise))
    try:
        if args.regen_stimuli:
            raise FileNotFoundError
        in_spikes_m = torch.load(os.path.join('synthetic_data', file_spikes), map_location=parameters['device'])
        labels_combined = torch.load(os.path.join('synthetic_data', file_labels_combined),
                                     map_location=parameters['device'])
    except FileNotFoundError:
        print('Generating new stimuli')
        if freqs2 is None:
            in_spikes_m,_, labels_combined = define_one_freq(freqs1, args.trials_n, args.noise, args.shift, parameters,
                                                             sprinkle_mag=args.sprinkle)
        else:
            in_spikes_m, labels_combined = define_two_freqs(freqs1, freqs2, args.trials_n, args.noise, args.shift,
                                                            parameters, args.sprinkle)
        labels_combined = torch.tensor(labels_combined)
        torch.save(in_spikes_m, os.path.join('synthetic_data', file_spikes))
        torch.save(labels_combined, os.path.join('synthetic_data', file_labels_combined))
    # print('in_spikes_m.shape', in_spikes_m.shape)
    # if in_spikes_m.is_sparse == False:
    #     in_spikes_m = in_spikes_m.to_sparse()
    # in_spikes_m_noneuron = in_spikes_m.to_dense()[:,:,0]
    # aaa = torch.where(in_spikes_m_noneuron)
    # plt.scatter(aaa[0], aaa[1], s=0.1)
    # plt.show()
    parameters['trials_per_stimulus'] = in_spikes_m.shape[1]
    batch_size = int(args.batch_size)
    probability_dist_labels = torch.zeros((labels_combined.shape[0], torch.unique(labels_combined).shape[0]))
    for i in range(labels_combined.shape[0]):
        probability_dist_labels[i, labels_combined[i].to('cpu').to(torch.int)] += 1
    if freqs2 is not None:
        labels_combined_str = [str([label.item() for label in labels]) for labels in labels_combined]
    else:
        labels_combined_str = [str(label.item()) for label in labels_combined]
    labels_combined_idx = torch.tensor(np.unique(labels_combined_str, return_inverse=True)[1])

    spll_layer = sPLL(parameters)
    dataset = torch.utils.data.TensorDataset(in_spikes_m.permute(1, 0, 2),labels_combined_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = tqdm(range(len(dataloader)), desc='Batches', disable=args.tqdm_silence, position=1, leave=False)
    time_bar = tqdm(range(1), desc='Time', disable=args.tqdm_silence, position=2, leave=False)

    ypred = []
    ylabel = []
    loss_batches = []
    batches.reset()
    for b_idx, (xlocal, ylocal) in enumerate(dataloader):
        batches.update(1)
        parameters['trials_per_stimulus'] = xlocal.shape[0]
        xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
        ylocal = ylocal.to(parameters['device'])
        spll_layer.initialize_state(parameters)
        spk_list = []
        times = range(xlocal.shape[0])
        time_bar.tot = xlocal.shape[0]
        time_bar.reset()
        for t in times:
            time_bar.update(1)
            spk = spll_layer(xlocal[t, :, :])[0]
            spk_list.append(spk.to_sparse())
        spk = torch.stack(spk_list, dim=0).to_dense()
        ypred.append(spk)
        ylabel.append(ylocal)
    ypred = torch.cat(ypred, dim=1)
    ylabel = torch.cat(ylabel, dim=0)

    file_spikes = 'spll_output-' + freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + '-' + sprinkle_desc + '_' + sim_time_desc + '-' + seed_desc + '.pt'
    file_labels_combined = 'spll_labels-' + freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + sprinkle_desc + '-' + sim_time_desc + '-' + seed_desc + '.pt'
    torch.save(ypred, file_spikes)
    torch.save(ylabel, file_labels_combined)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Encoding")
    parser.add_argument("--seed", type=int, default=6, help="Random seed. Default: -1")
    parser.add_argument("--sim_time", type=float, default=0.1, help="Simulation Time (s)")
    parser.add_argument("--trials_n", type=int, default=100, help="Trials per stimulus")
    parser.add_argument("--noise", type=float, default=0.1, help="Jitter noise (% of freq)")
    parser.add_argument("--shift", type=float, default=1, help="Random shift (% of freq)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Random shift (% of freq)")
    parser.add_argument("--gpu", action='store_true', help='Use GPU if available')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr_spikes", type=float, default=0.1, help="Learning rate Spikes")
    parser.add_argument("--lr_dec", type=float, default=1, help="Learning rate Decoder")
    parser.add_argument('--clock_sim', type=float, default=1e-3, help='Simulation clock (s)')
    parser.add_argument('--regen_stimuli', action='store_true', help='Regenerate stimuli')
    parser.add_argument('--nni_opt', action='store_true', help='NNI optimization')
    parser.add_argument('--tqdm_silence', action='store_true', help='Silence tqdm')
    parser.add_argument('--thr_acc', type=float, default=0.8, help='Threshold for accuracy')
    parser.add_argument('--train_spll', action='store_true', help='Train SPLL parameters')
    parser.add_argument('--figures', type=int, default=0,
                        help='Every when save figures in Tensorboard, put 0 for never (slow)')
    parser.add_argument('--load_optimal', action='store_true', help='Load optimal parameters')
    parser.add_argument('--early_stop', action='store_true', help='Early stop')
    parser.add_argument('--sprinkle', type=float, default=0, help='Sprinkle some random spikes in the dataset')

    parser.add_argument('--neurons_n', type=int, default=50, help='Number of neurons in the layer')
    parser.add_argument('--C_Osc', type=float, default=1, help='Oscillator capacitance')
    parser.add_argument('--v_Osc_threshold', type=float, default=1, help='Oscillator threshold')
    parser.add_argument('--tau_Osc_Ie', type=float, default=100e-3, help='Oscillator time constant for Ie')
    parser.add_argument('--tau_Osc', type=float, default=100e-3, help='Oscillator time constant')
    parser.add_argument('--reset_osc', type=float, default=0, help='Oscillator reset potential')
    parser.add_argument('--I_minimum_osc', type=float, default=16, help='Oscillator minimum current')
    parser.add_argument('--I_step_osc', type=float, default=2, help='Oscillator step current')
    parser.add_argument('--refrac_Osc', type=float, default=1e-3, help='Oscillator refractory period')
    parser.add_argument('--TDE_to_Osc_current', type=float, default=1., help='TDE to Oscillator current')

    parser.add_argument('--C_TDE', type=float, default=1, help='TDE capacitance')
    parser.add_argument('--v_TDE_threshold', type=float, default=1, help='TDE threshold')
    parser.add_argument('--tau_TDE', type=float, default=10e-3, help='TDE time constant')
    parser.add_argument('--reset_TDE', type=float, default=0, help='TDE reset potential')
    parser.add_argument('--tau_trg_TDE', type=float, default=1e-3, help='TDE time constant for trigger')
    parser.add_argument('--tau_fac_TDE', type=float, default=5e-3, help='TDE time constant for facilitatory')
    parser.add_argument('--gain_trg_TDE', type=float, default=1, help='TDE gain fro trigger')
    parser.add_argument('--gain_fac_TDE', type=float, default=1, help='TDE gain for facilitatory')
    parser.add_argument('--refrac_TDE', type=float, default=1e-3, help='TDE refractory period')

    args = parser.parse_args()
    main(args)



