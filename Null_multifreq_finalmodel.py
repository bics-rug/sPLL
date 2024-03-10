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

LOG = logging.getLogger('snn')
from utils import *
from models import *
from sklearn.metrics import accuracy_score




# def accuracy(spiking_net,decoder,parameters_enc,parameters,test_dataloader,writer,epoch,args):
#     ypred_lif = []
#     ylabel = []
#     batches = tqdm(range(len(test_dataloader)), desc='Batches', disable=False)
#     time_bar = tqdm(range(1), desc='Time', disable=False)
#     batches.reset()
#     with torch.no_grad():
#         for b_idx, (xlocal, ylocal) in enumerate(test_dataloader):
#             batches.update(1)
#             parameters_enc['trials_per_stimulus'] = xlocal.shape[0]
#             parameters['trials_per_stimulus'] = xlocal.shape[0]
#             xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
#             ylocal = ylocal.to(parameters['device'])
#             for layer in spiking_net:
#                 layer.initialize_state(parameters_enc)
#             spk_enc_list = []
#             times = range(xlocal.shape[0])
#             time_bar.tot = xlocal.shape[0]
#             time_bar.reset()
#             for t in times:
#                 time_bar.update(1)
#                 spk_enc = spiking_net(xlocal[t,:,0][:,None])
#                 spk_enc_list.append(spk_enc.to_sparse())
#             spk_enc_lif = torch.stack(spk_enc_list, dim=0).to_dense()
#             spk_enc_lif_sum = spk_enc_lif.sum(dim=0)# / basic_freqs
#             lif_out = decoder(spk_enc_lif_sum.to(torch.float32))
#             ylabel.append(ylocal)
#             ypred_lif.append(lif_out)
#         ypred_lif = torch.concatenate(ypred_lif)
#         ylabel = torch.concatenate(ylabel)
#         if (args.figures>0):
#             if (epoch % args.figures == 0):
#                 fig1, axis1 = plt.subplots(ncols=2, nrows=2)
#                 im0 = axis1[0, 0].imshow(torch.sum(xlocal, dim=0).cpu(), aspect='auto', cmap='Greens', interpolation='None')
#                 divider = make_axes_locatable(axis1[0, 0])
#                 cax = divider.append_axes('right', size='5%', pad=0.05)
#                 fig1.colorbar(im0, cax=cax, orientation='vertical')
#                 axis1[0, 0].set_title('xlocal')
#                 im1 = axis1[0, 1].imshow(spk_enc_lif_sum.cpu().detach().numpy(), aspect='auto', cmap='Purples',
#                                          interpolation='None')
#                 divider = make_axes_locatable(axis1[0, 1])
#                 cax = divider.append_axes('right', size='5%', pad=0.05)
#                 fig1.colorbar(im1, cax=cax, orientation='vertical')
#                 axis1[0, 1].set_title('spk_enc_lif_sum')
#                 im2 = axis1[1, 0].imshow((torch.abs(ypred_lif-1)<parameters['thr_acc']).to(int).cpu().detach().numpy(), aspect='auto', cmap='Reds', interpolation='None')
#                 divider = make_axes_locatable(axis1[1, 0])
#                 cax = divider.append_axes('right', size='5%', pad=0.05)
#                 fig1.colorbar(im2, cax=cax, orientation='vertical')
#                 axis1[1, 0].set_title('ypred_lif>'+str(parameters['thr_acc']))
#                 im3 = axis1[1, 1].imshow(ylabel.cpu().detach().numpy(), aspect='auto', cmap='Blues', interpolation='None')
#                 divider = make_axes_locatable(axis1[1, 1])
#                 cax = divider.append_axes('right', size='5%', pad=0.05)
#                 fig1.colorbar(im3, cax=cax, orientation='vertical')
#                 axis1[1, 1].set_title('ylabel')
#                 fig1.tight_layout()
#                 writer.add_figure('Test', fig1, epoch)
#                 fig2, axis2 = plt.subplots(nrows=2)
#                 events_in = torch.where(xlocal[:,:,0])
#                 axis2[0].scatter(events_in[0].cpu(), events_in[1].cpu(), s=0.1)
#                 events_out = torch.where(spk_enc_lif.flatten(1,2))
#                 axis2[1].scatter(events_out[0].cpu(), events_out[1].cpu(), s=0.1)
#                 fig2.tight_layout()
#                 writer.add_figure('Test_spikes', fig2, epoch)
#
#         # print(ypred_lif[0])
#         # print((np.abs(ypred_lif-1)<parameters['thr_acc'])[0].to(int))
#         # print(ylabel[0].to(int))
#         # loss_lif = nn.MSELoss()(ypred_lif, ylabel)
#         # print(loss_lif)
#         cm_lif = multilabel_confusion_matrix(ylabel.cpu().detach().numpy(),(ypred_lif>parameters['thr_acc']).to(int).cpu().detach().numpy())
#         acc = []
#         for b in range(cm_lif.shape[0]):
#             acc.append((cm_lif[b,0,0]+cm_lif[b,1,1])/cm_lif[b].sum())
#         return float(np.mean(acc))
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Encoding")
    parser.add_argument("--seed", type=int, default=6, help="Random seed. Default: -1")
    parser.add_argument("--sim_time", type=float, default=0.1, help="Simulation Time (s)")
    parser.add_argument("--trials_n", type=float, default=100, help="Trials per stimulus")
    parser.add_argument("--noise", type=float, default=100, help="Jitter noise (% of freq)")
    parser.add_argument("--shift", type=float, default=1, help="Random shift (% of freq)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Random shift (% of freq)")
    parser.add_argument("--gpu", action='store_true',help='Use GPU if available')
    parser.add_argument("--epochs",type=int,default=100,help="Number of epochs")
    parser.add_argument("--lr_spikes",type=float,default=0.1,help="Learning rate Spikes")
    parser.add_argument("--lr_dec",type=float,default=1,help="Learning rate Decoder")
    parser.add_argument('--clock_sim',type=float,default=1e-3,help='Simulation clock (s)')
    parser.add_argument('--regen_stimuli',action='store_true',help='Regenerate stimuli')
    parser.add_argument('--nni_opt',action='store_true',help='NNI optimization')
    parser.add_argument('--tqdm_silence',action='store_true',help='Silence tqdm')
    parser.add_argument('--thr_acc',type=float,default=0.8,help='Threshold for accuracy')
    parser.add_argument('--train_spll',action='store_true',help='Train SPLL parameters')
    parser.add_argument('--figures',type=int,default=0,help='Every when save figures in Tensorboard, put 0 for never (slow)')
    parser.add_argument('--load_optimal',action='store_true',help='Load optimal parameters')
    parser.add_argument('--early_stop',action='store_true',help='Early stop')
    parser.add_argument('--sprinkle',type=float,default=0,help='Sprinkle some random spikes in the dataset')



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
    parameters = vars(args)  # copy by reference (checked below)

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
        opt_params = json.load(open('SPLL_multifreq_opt.txt', 'r'))
        print('Loading opt params', opt_params)
        for key, val in opt_params.items():
            parameters[key] = val
            LOG.debug(print(key))
            assert (args.__dict__[key] == parameters[key])
    parameters['n_in'] = 1
    print(parameters)
    np.random.seed(args.seed)
    freqs1 = np.arange(30, 57, 3)
    freqs2 = np.arange(30, 57, 9)
    # parameters = {}
    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    print('device in use:', parameters['device'])
    _,test_dataloader,labels_combined_str = import_dataloaders(freqs1, freqs2, args, parameters)

    output_n = len(np.unique(labels_combined_str))
    decoder = nn.Linear(1, output_n,bias=False).to(parameters['device'])
    decoder.load_state_dict(torch.load(os.path.join(os.getcwd(), 'seed_coll', f'Null_multifreq_decoder_6.pt')))
    writer = SummaryWriter()
    epoch = 0

    batches = tqdm(range(len(test_dataloader)), desc='Batches', disable=False)
    time_bar = tqdm(range(1), desc='Time', disable=False)
    batches.reset()
    ypred = []
    ylabel = []
    CCO_spike_list = []
    TDE_spike_list = []
    with torch.no_grad():
        for b_idx, (xlocal, ylocal) in enumerate(test_dataloader):
            batches.update(1)
            xlocal = xlocal.permute(1, 0, 2).to(parameters['device'])
            ylocal = ylocal.to(parameters['device'])
            parameters['trials_per_stimulus'] = xlocal.shape[1]
            lif_out = decoder(xlocal[:,:,0].sum(dim=0)[:,None].to(torch.float32))
            ypred.append(lif_out.argmax(dim=1))
            ylabel.append(ylocal)
    ypred = torch.cat(ypred)
    ylabel = torch.cat(ylabel)
    acc = accuracy_score(ypred,ylabel.cpu())
    mydict = {'spike_mean': torch.nan, 'spike_std': torch.nan, 'acc': acc}
    torch.save(mydict, os.path.join(os.getcwd(), 'seed_coll', f'SPLL_multifreq_spikes_{args.seed}.pt'))
    print('spk_enc_sum_cumulative: mean', torch.nan, 'std', torch.nan,'acc',acc)
    print('ciao')