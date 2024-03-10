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
import json
from sklearn.metrics import accuracy_score
LOG = logging.getLogger('snn')
from utils import *
from models import *
from torchviz import make_dot





def accuracy(spiking_net,decoder,parameters_enc,parameters,test_dataloader,writer,epoch,args):
    ypred_lif = []
    ylabel = []
    batches = tqdm(range(len(test_dataloader)), desc='Batches', disable=True)
    time_bar = tqdm(range(1), desc='Time', disable=True)
    batches.reset()
    spk_coll = []
    spk_count_coll = []
    with torch.no_grad():
        for b_idx, (xlocal, ylocal) in enumerate(test_dataloader):
            batches.update(1)
            parameters_enc['trials_per_stimulus'] = xlocal.shape[0]
            parameters['trials_per_stimulus'] = xlocal.shape[0]
            xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
            ylocal = ylocal.to(parameters['device'])
            for layer in spiking_net:
                layer.initialize_state(parameters_enc)
            spk_enc_list = []
            times = range(xlocal.shape[0])
            time_bar.tot = xlocal.shape[0]
            time_bar.reset()
            xlocal = xlocal.coalesce()
            spikes = torch.zeros((xlocal.shape[1])).to(parameters['device'])
            # print('ylocal test',ylocal)
            spk_count = 0
            for t in times:
                spikes *= 0
                time_idx = xlocal.indices()[0]
                trial_idx = xlocal.indices()[1]
                neurons_idx = xlocal.indices()[2]
                time_idx_s = (time_idx == t)
                neurons_idx_s = (neurons_idx == 0)
                sel = torch.where(time_idx_s & neurons_idx_s)
                spikes[trial_idx[sel]] = 1
                spk_enc = spiking_net(spikes[:, None])
                # print(spk_enc.to_sparse())
                for layer in spiking_net:
                    spk_count = spk_count + torch.sum(layer.state.spk,dim=0)
                    print('layer.state.spk',torch.sum(layer.state.spk))
                print('spk_count',spk_count)
                spk_enc_list.append(spk_enc.to_sparse())
                time_bar.update(1)
            spk_count_coll.append(spk_count)
            spk_enc_lif = torch.stack(spk_enc_list, dim=0).to_dense()
            spk_enc_lif_sum = spk_enc_lif.sum(dim=0)# / basic_freqs
            # print('spk_enc_lif_sum',spk_enc_lif_sum)
            lif_out = decoder(spk_enc_lif_sum.to(torch.float32))
            # print('lif_out',lif_out)
            xwinner = torch.argmax(lif_out, dim=1)
            ylabel.append(ylocal)
            ypred_lif.append(xwinner)
            spk_coll.append(spk_enc_lif_sum)
            # print('lif_out_max', torch.argmax(lif_out, dim=1))
            # print('ylocal', ylocal)
            # print('accuracy', torch.sum(torch.argmax(lif_out, dim=1) == ylocal).float() / ylocal.shape[0])
        ypred_lif = torch.concatenate(ypred_lif)
        ylabel = torch.concatenate(ylabel)
        #print(spk_coll)
        spk_count = torch.mean(torch.concatenate(spk_count_coll,dim=0))
        if (args.figures>0):
            if (epoch % args.figures == 0):
                # fig1, axis1 = plt.subplots(ncols=2, nrows=2)
                # im0 = axis1[0, 0].imshow(torch.sum(xlocal, dim=0).to_dense().cpu(), aspect='auto', cmap='Greens', interpolation='None')
                # divider = make_axes_locatable(axis1[0, 0])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig1.colorbar(im0, cax=cax, orientation='vertical')
                # axis1[0, 0].set_title('xlocal')
                # im1 = axis1[0, 1].imshow(spk_enc_lif_sum.cpu().detach().numpy(), aspect='auto', cmap='Purples',
                #                          interpolation='None')
                # divider = make_axes_locatable(axis1[0, 1])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig1.colorbar(im1, cax=cax, orientation='vertical')
                # axis1[0, 1].set_title('spk_enc_lif_sum')
                # im2 = axis1[1, 0].imshow((torch.abs(ypred_lif-1)<parameters['thr_acc']).to(int).cpu().detach().numpy(), aspect='auto', cmap='Reds', interpolation='None')
                # divider = make_axes_locatable(axis1[1, 0])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig1.colorbar(im2, cax=cax, orientation='vertical')
                # axis1[1, 0].set_title('ypred_lif>'+str(parameters['thr_acc']))
                # im3 = axis1[1, 1].imshow(ylabel.cpu().detach().numpy(), aspect='auto', cmap='Blues', interpolation='None')
                # divider = make_axes_locatable(axis1[1, 1])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig1.colorbar(im3, cax=cax, orientation='vertical')
                # axis1[1, 1].set_title('ylabel')
                # fig1.tight_layout()
                # writer.add_figure('Test', fig1, epoch)
                fig2, axis2 = plt.subplots(nrows=2)
                time_idx = xlocal.indices()[0]
                trial_idx = xlocal.indices()[1]
                neuron_idx = xlocal.indices()[2]
                neuron_idx_s = (neuron_idx == 0)
                # print(xlocal.shape)
                # print(trial_idx[neuron_idx_s])
                axis2[0].scatter(time_idx[neuron_idx_s].cpu(), trial_idx[neuron_idx_s].cpu(), s=0.1)
                events_out = torch.where(spk_enc_lif.flatten(1,2))
                axis2[1].scatter(events_out[0].cpu(), events_out[1].cpu(), s=0.1)
                fig2.tight_layout()
                writer.add_figure('Test_spikes', fig2, epoch)
        # print('spk_sum',spk_coll.sum())
        # print('spk',spk_coll[0])

        # print('ypred_sum',ypred_lif.sum())
        # print('ypred',ypred_lif[0])
        # print((np.abs(ypred_lif-1)<parameters['thr_acc'])[0].to(int))
        # print('ylabel sum',ylabel.sum().to(int))
        # print('ylabel',ylabel)
        # loss_lif = nn.MSELoss()(ypred_lif, ylabel)
        # print(loss_lif)
        # softmax = nn.Softmax(dim=1)
        # ypred_test = ypred_lif/ypred_lif.max(dim=1, keepdim=True)[0]
        # print('ypred',ypred_lif)
        # print('ypred>thr_acc sum',(ypred_lif>parameters['thr_acc']).sum().to(int))
        # print('ypred>thr_acc',(ypred_lif>parameters['thr_acc'])[0].to(int))

        acc = accuracy_score(ypred_lif.cpu(),ylabel.cpu())
        # acc = accuracy_score(ylabel.cpu().detach().numpy(), (ypred_lif>parameters['thr_acc']).cpu().detach().numpy())
        print('acc',acc)
        print('spk_count',spk_count)
        return acc,spk_count
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
    parser.add_argument('--tau_Osc_Irec',type=float,default=1e-3,help='Encoder Layer Synaptic Tau (s)')
    parser.add_argument('--tau_Osc_Ie',type=float,default=1e-3,help='Encoder Layer Synaptic Tau (s)')
    parser.add_argument('--tau_Osc',type=float,default=10e-3,help='Encoder LayerMembrane Tau (s)')
    parser.add_argument('--neurons_n',type=int,default=50,help='Number of neurons in the layer')
    parser.add_argument('--enc_w_in_mean',type=float,default=0,help='Encoder Layer Mean Input Weight (adim)')
    parser.add_argument('--enc_w_in_std',type=float,default=1,help='Encoder Layer Std Input Weight (adim)')
    parser.add_argument('--gain_syn',type=float,default=1,help='Gain Synapse (adim)')
    parser.add_argument('--gain_syn_rec',type=float,default=0.1,help='Gain Synapse Recurrent (adim)')
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

    parameters['n_layers'] = len(args.layers_size)

    parameters['C_Osc'] = 1e-3
    parameters['v_Osc_threshold'] = 1

    parameters['reset_osc'] = 0
    parameters['I_minimum_osc'] = 0
    parameters['I_step_osc'] = 0
    parameters['refrac_Osc'] = 1e-3

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
        opt_params = json.load(open(f'LIF_multifreq_{args.layers_size}_opt.txt', 'r'))
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


    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    # freqs2 = None
    # parameters = {}
    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    print('device in use:', parameters['device'])
    train_dataloader, test_dataloader, labels_combined_str = import_dataloaders(freqs1, freqs2, args, parameters,sparse=True)

    torch.manual_seed(args.seed)



    parameters_enc = parameters.copy()

    parameters_enc['weight_mean'] = parameters['enc_w_in_mean']
    parameters_enc['weight_std'] = parameters['enc_w_in_std']

    # basic_freqs = torch.load('basic_freqs.pt')
    # basic_freqs = torch.tensor(basic_freqs)
    # basic_freqs = basic_freqs.to(parameters['device'])


    output_n = len(np.unique(labels_combined_str))
    print('Number of classes:', len(np.unique(labels_combined_str)))
    layers = []
    parameters_here = parameters_enc.copy()
    parameters_here['n_in'] = 1
    parameters_here['neurons_n'] = args.layers_size[0]
    layers.append(LIF_neuron(parameters_here, train=True, recurrent=True).to(parameters['device']))
    for layer in range(parameters['n_layers']-1):
        parameters_here['n_in'] = args.layers_size[layer]
        parameters_here['neurons_n'] = args.layers_size[layer+1]
        layers.append(LIF_neuron(parameters_here,train=True,recurrent=True).to(parameters['device']))
    # layers.append(LIF_neuron(parameters_here,train=True,recurrent=True).to(parameters['device']))
    # parameters_here = parameters_enc.copy()
    # parameters_here['n_in'] = args.layers_size[0]
    # parameters_here['neurons_n'] = args.layers_size[1]
    # layers.append(LIF_neuron(parameters_here,train=True,recurrent=True).to(parameters['device']))
    # parameters_here = parameters_enc.copy()
    # parameters_here['n_in'] = args.layers_size[1]
    # parameters_here['neurons_n'] = args.layers_size[2]
    # layers.append(LIF_neuron(parameters_here,train=True,recurrent=True).to(parameters['device']))
    spiking_net = nn.Sequential(*layers)
    spiking_net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'seed_coll', f'LIF_multifreq_{args.layers_size}_net_{args.seed}.pt')))

    decoder = nn.Linear(args.layers_size[-1], output_n).to(parameters['device'])
    decoder.load_state_dict(torch.load(os.path.join(os.getcwd(), 'seed_coll', f'LIF_multifreq_{args.layers_size}_decoder_{args.seed}.pt')))

    epochs = tqdm(range(args.epochs), desc='Epochs', disable=args.tqdm_silence, position=0, leave=True)
    batches = tqdm(range(len(train_dataloader)), desc='Batches', disable=args.tqdm_silence, position=1, leave=False)
    time_bar = tqdm(range(1), desc='Time', disable=args.tqdm_silence, position=2, leave=False)
    batches.total = len(train_dataloader)
    loss_coll = []
    acc_coll = []
    # softmax = nn.Softmax(dim=1)

    acc,spk_count = accuracy(spiking_net, decoder, parameters_enc, parameters, test_dataloader, None, 0, args)
    torch.save(spk_count, os.path.join(os.getcwd(), 'seed_coll', f'LIF_multifreq_{args.layers_size}_spk_count_{args.seed}.pt'))
    torch.save(acc, os.path.join(os.getcwd(), 'seed_coll', f'LIF_multifreq_{args.layers_size}_final_{args.seed}.pt'))
