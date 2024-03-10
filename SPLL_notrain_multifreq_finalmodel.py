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
from sklearn.metrics import accuracy_score
import nni
import logging
import os
import json

LOG = logging.getLogger('snn')
from utils import *
from models import *
import SPLL_dataset





def accuracy(spiking_net,decoder,parameters_enc,parameters,test_dataloader,writer,epoch,args):
    ypred_lif = []
    ylabel = []
    batches = tqdm(range(len(test_dataloader)), desc='Batches', disable=True)
    time_bar = tqdm(range(1), desc='Time', disable=True)
    batches.reset()
    with torch.no_grad():
        for b_idx, (xlocal, ylocal) in enumerate(test_dataloader):
            batches.update(1)
            xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
            ylocal = ylocal.to(parameters['device'])
            spk_sum = xlocal.sum(dim=0)
            #print(spk_sum.shape)
            lif_out = decoder(spk_sum.to(torch.float32))
            xwinner = torch.argmax(lif_out, dim=1)
            ylabel.append(ylocal)
            ypred_lif.append(xwinner)
        ypred_lif = torch.concatenate(ypred_lif)
        ylabel = torch.concatenate(ylabel)
        acc = accuracy_score(ypred_lif, ylabel)

        return float(np.mean(acc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Encoding")
    parser.add_argument("--seed", type=int, default=6, help="Random seed. Default: -1")
    parser.add_argument("--sim_time", type=float, default=0.1, help="Simulation Time (s)")
    parser.add_argument("--trials_n", type=int, default=100, help="Trials per stimulus")
    parser.add_argument("--noise", type=float, default=100, help="Jitter noise (% of freq)")
    parser.add_argument("--shift", type=float, default=1, help="Random shift (% of freq)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Random shift (% of freq)")
    parser.add_argument("--gpu", action='store_true',help='Use GPU if available')
    parser.add_argument("--epochs",type=int,default=100,help="Number of epochs")
    parser.add_argument("--lr_spikes",type=float,default=0.1,help="Learning rate Spiking")
    parser.add_argument("--lr_dec",type=float,default=1,help="Learning rate Decoder")
    parser.add_argument('--clock_sim',type=float,default=1e-3,help='Simulation clock (s)')
    parser.add_argument('--neurons_n',type=int,default=50,help='Number of neurons in the layer')
    parser.add_argument('--enc_w_in_mean',type=float,default=0,help='Encoder Layer Mean Input Weight (adim)')
    parser.add_argument('--enc_w_in_std',type=float,default=1,help='Encoder Layer Std Input Weight (adim)')
    parser.add_argument('--regen_stimuli',action='store_true',help='Regenerate stimuli')
    parser.add_argument('--layers_size',type=str,default='50,50',help='Number of neurons in each layer')
    parser.add_argument('--nni_opt',action='store_true',help='NNI optimization')
    parser.add_argument('--tqdm_silence',action='store_true',help='Silence tqdm')
    parser.add_argument('--thr_acc',type=float,default=0.8,help='Threshold for accuracy')
    parser.add_argument('--figures',type=int,default=0,help='Every when save figures in Tensorboard, put 0 for never (slow)')
    parser.add_argument('--load_optimal',action='store_true',help='Load optimal parameters')
    parser.add_argument('--early_stop',action='store_true',help='Early stop')
    parser.add_argument('--reg_l0',type=float,default=0,help='L0 regularization')
    parser.add_argument('--reg_l1',type=float,default=0,help='L1 regularization')
    parser.add_argument('--avg_spike_count',type=int,default=20,help='Regularization to Average spike count')
    parser.add_argument('--sprinkle',type=float,default=0,help='Sprinkle some random spikes in the dataset')
    parser.add_argument('--f1_start',type=float,default=30,help='Start frequency1')
    parser.add_argument('--f1_end',type=float,default=57,help='End frequency1')
    parser.add_argument('--f1_step',type=float,default=3,help='Step frequency1')
    parser.add_argument('--f2_start',type=float,default=30,help='Start frequency2')
    parser.add_argument('--f2_end',type=float,default=57,help='End frequency2')
    parser.add_argument('--f2_step',type=float,default=9,help='Step frequency2')

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
    print(args.layers_size)
    if type(args.layers_size) is str:
        args.layers_size = list(map(int,args.layers_size.split(',')))
        print(args.layers_size)
    parameters = vars(args)  # copy by reference (checked below)

    parameters['n_layers'] = 2
    if args.nni_opt:
        # Get new set of params
        PARAMS = nni.get_next_parameter()
        print(PARAMS)
        # Replace default args with new set
        for key, val in PARAMS.items():
            parameters[key] = val
            LOG.debug(print(key))
            assert (args.__dict__[key] == parameters[key])
    

    if args.load_optimal:
        opt_params = json.load(open('LIF_multifreq_opt.txt', 'r'))
        print('Loading opt params',opt_params)
        for key, val in opt_params.items():
            parameters[key] = val
            LOG.debug(print(key))
            assert (args.__dict__[key] == parameters[key])
        # layer_size = []
        # for layer in range(parameters['n_layers']):
        #     key = 'n_neurons_l' + str(layer)
        #     layer_size.append(parameters[key])
        # args.layers_size = layer_size

    print(parameters)

    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    print('device in use:', parameters['device'])


    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    # parameters = {}
    print('what')
    tmp = args.tqdm_silence
    args.tqdm_silence = True
    SPLL_dataset.main(args)
    args.tqdm_silence = tmp
    name = gen_name(freqs1, freqs2, args, parameters)
    spikes_spll = torch.load(os.path.join('synthetic_data', 'spll_output-' + name + '.pt'))
    spikes_spll_sum = torch.sum(spikes_spll, dim=0)

    labels_combined_str = torch.load(os.path.join('synthetic_data', 'spll_labels-' + name + '.pt'))

    X_train, X_test, y_train, y_test = train_test_split(spikes_spll.permute(1,0,2).to('cpu'),
                                                        labels_combined_str, test_size=0.1,
                                                        random_state=args.seed, stratify=labels_combined_str)

    if args.nni_opt:
        X_train, X_test, y_train, y_test = train_test_split(X_train.to('cpu'),
                                                            y_train, test_size=0.1, random_state=args.seed,stratify=y_train)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    torch.manual_seed(args.seed)
    parameters_enc = parameters.copy()

    parameters_enc['weight_mean'] = parameters['enc_w_in_mean']
    parameters_enc['weight_std'] = parameters['enc_w_in_std']

    # basic_freqs = torch.load('basic_freqs.pt')
    # basic_freqs = torch.tensor(basic_freqs)
    # basic_freqs = basic_freqs.to(parameters['device'])


    output_n = len(np.unique(labels_combined_str))
    decoder = nn.Linear(args.neurons_n, output_n,bias=False).to(parameters['device'])
    decoder.load_state_dict(torch.load(os.path.join(os.getcwd(),'seed_coll',f'SPLL_multifreq_decoder_{args.seed}.pt')))
    if args.nni_opt:
        log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
        comment = None
    else:
        log_dir = None
        comment = 'SPLL_multifreq'
    writer = SummaryWriter(log_dir=log_dir, comment=comment)
    criterion_lif = nn.CrossEntropyLoss()
    optimizing_params = [{'params': decoder.parameters(),'lr':args.lr_dec}]
    optimizer_lif = torch.optim.Adamax(optimizing_params)
    epochs = tqdm(range(args.epochs), desc='Epochs', disable=args.tqdm_silence, position=0, leave=True)
    batches = tqdm(range(len(train_dataloader)), desc='Batches', disable=args.tqdm_silence, position=1, leave=False)
    time_bar = tqdm(range(1), desc='Time', disable=args.tqdm_silence, position=2, leave=False)
    batches.total = len(train_dataloader)
    loss_coll = []
    acc_coll = []
    acc = accuracy(None, decoder, parameters_enc, parameters, test_dataloader, writer, 0, args)

    try:
        os.mkdir(os.path.join(os.getcwd(),'seed_coll'))
    except FileExistsError:
        pass
    torch.save(acc,os.path.join(os.getcwd(),'seed_coll',f'SPLL_multifreq_acc_final_model_{args.seed}.pt'))
