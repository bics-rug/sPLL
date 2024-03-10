import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
# from bokeh.plotting import figure, show, row, output_notebook
# from bokeh.plotting import output_notebook
# output_notebook()
import tqdm
import nni
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
            spk_sum = xlocal[:, :, 0].sum(dim=0)[:, None]
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
    parser.add_argument('--tau_Osc_Ie',type=float,default=1e-3,help='Encoder Layer Synaptic Tau (s)')
    parser.add_argument('--tau_Osc',type=float,default=1e-3,help='Encoder LayerMembrane Tau (s)')
    parser.add_argument('--neurons_n',type=int,default=50,help='Number of neurons in the layer')
    parser.add_argument('--enc_w_in_mean',type=float,default=0,help='Encoder Layer Mean Input Weight (adim)')
    parser.add_argument('--enc_w_in_std',type=float,default=1,help='Encoder Layer Std Input Weight (adim)')
    parser.add_argument('--gain_syn',type=float,default=1000,help='Gain Synapse (adim)')
    parser.add_argument('--gain_syn_rec',type=float,default=1000,help='Gain Synapse Recurrent (adim)')
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

    args = parser.parse_args()
    print(args.layers_size)
    if type(args.layers_size) is str:
        args.layers_size = list(map(int,args.layers_size.split(',')))
        print(args.layers_size)
    parameters = vars(args)  # copy by reference (checked below)
    parameters['C_Osc'] = 1
    parameters['v_Osc_threshold'] = 1

    parameters['reset_osc'] = 0
    parameters['I_minimum_osc'] = 0
    parameters['I_step_osc'] = 0
    parameters['refrac_Osc'] = 1e-3

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
    parameters['TDE_to_Osc_current'] = parameters['gain_syn']
    print(parameters)
    parameters['n_layers'] = 2

    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    print('device in use:', parameters['device'])


    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    # parameters = {}

    train_dataloader, test_dataloader, labels_combined_str = import_dataloaders(freqs1,freqs2,args,parameters)
    torch.manual_seed(args.seed)
    parameters_enc = parameters.copy()

    parameters_enc['weight_mean'] = parameters['enc_w_in_mean']
    parameters_enc['weight_std'] = parameters['enc_w_in_std']

    # basic_freqs = torch.load('basic_freqs.pt')
    # basic_freqs = torch.tensor(basic_freqs)
    # basic_freqs = basic_freqs.to(parameters['device'])


    output_n = len(np.unique(labels_combined_str))
    decoder = nn.Linear(1, output_n,bias=True).to(parameters['device'])
    if args.nni_opt:
        log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
        comment = None
    else:
        log_dir = None
        comment = 'Null_multifreq'
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
    softmax = nn.Softmax(dim=1)
    for epoch in epochs:
        ypred_lif = []
        ylabel = []
        loss_lif_batches = []
        batches.reset()
        for b_idx, (xlocal, ylocal) in enumerate(train_dataloader):
            batches.update(1)
            xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
            ylocal = ylocal.to(parameters['device'])

            spk_sum = xlocal[:,:,0].sum(dim=0).reshape(-1,1)
            # events = xlocal[:,:,0].nonzero(as_tuple=True)
            # plt.scatter(events[0], events[1], s=0.1)

            lif_out = decoder(spk_sum.to(torch.float32))
            # plt.figure()
            # plt.imshow(lif_out.detach().cpu().numpy(), aspect='auto')
            # print(lif_out)
            # print(ylocal)
            #
            # plt.show()
            # # print(lif_out.shape)
            # out_soft = softmax(lif_out)
            ylabel.append(ylocal)
            ypred_lif.append(lif_out)
            loss_lif = criterion_lif(lif_out , ylocal)
            optimizer_lif.zero_grad()
            loss_lif.backward()
            loss_lif_batches.append(loss_lif.item())
            optimizer_lif.step()
        ypred_lif = torch.concatenate(ypred_lif)
        ylabel = torch.concatenate(ylabel)
        writer.add_scalar("Loss/train_lif", np.mean(loss_lif_batches), epoch)
        writer.add_histogram("Weights_l0", decoder.weight, epoch)
        acc = accuracy(None,decoder,parameters_enc,parameters,test_dataloader,writer,epoch,args)
        epochs.set_postfix_str(
            f"loss: {np.mean(loss_lif_batches):.3f} acc: {acc:.3f}")
        loss_coll.append(np.mean(loss_lif_batches))
        acc_coll.append(acc)
        writer.add_scalar("Accuracy", acc, epoch)
        nni.report_intermediate_result(acc)
    acc = accuracy(None, decoder, parameters_enc, parameters, test_dataloader, writer, epoch, args)
    nni.report_final_result(acc)
    try:
        os.mkdir(os.path.join(os.getcwd(),'seed_coll'))
    except FileExistsError:
        pass
    torch.save(torch.tensor(acc_coll),os.path.join(os.getcwd(),'seed_coll',f'Null_multifreq_acc_coll_{args.seed}.pt'))
    torch.save(decoder.state_dict(),os.path.join(os.getcwd(),'seed_coll',f'Null_multifreq_decoder_{args.seed}.pt'))
