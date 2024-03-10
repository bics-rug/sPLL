import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
# from bokeh.plotting import figure, show, row, output_notebook
# from bokeh.plotting import output_notebook
# output_notebook()
from tqdm import trange# import seaborn as sns
from matplotlib.colors import ListedColormap
import itertools
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skmultilearn.model_selection import iterative_train_test_split
import nni
import logging
import os
import json
from utils import *
from models import *
LOG = logging.getLogger('snn')
import re
from sklearn.metrics import accuracy_score
def fft_dataset(args,parameters):
    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    if args.onefreq:
        freqs2 = None
    else:
        freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    in_spikes_m, labels_combined_str = import_dataloaders(freqs1, freqs2, args, parameters, sparse=True,
                                                          return_datasets=True)
    rate = 1 / args.clock_sim
    fmax = 100
    in_spikes = in_spikes_m[:, :, 0]
    fft_coll = []
    for trial in range(in_spikes.shape[0] - 1, -1, -1):
        datafreqs = np.fft.rfftfreq(in_spikes.shape[0],d=1/rate)
        index_fmax = np.where(datafreqs<fmax)[0]
        fft = np.real(np.fft.rfft(in_spikes[trial, :], in_spikes.shape[0])).flatten()
        fft_coll.append(fft[index_fmax])
    return fft_coll,labels_combined_str
def isi_dataset(args,parameters):
    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    if args.onefreq:
        freqs2 = None
    else:
        freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    in_spikes_m, labels_combined_str = import_dataloaders(freqs1, freqs2, args, parameters, sparse=True,
                                                          return_datasets=True)
    rate = 1 / args.clock_sim
    fmax = 100
    in_spikes = in_spikes_m[:, :, 0]
    events = torch.where(in_spikes.T == 1)
    isi_coll = []
    for trial in range(in_spikes.shape[0] - 1, -1, -1):
        datafreqs = np.fft.rfftfreq(in_spikes.shape[0],d=1/rate)
        index_fmax = np.where(datafreqs<fmax)[0]
        isis = np.diff(events[0][events[1]==trial])
        hist = np.histogram(rate/isis,bins=index_fmax.shape[0],range=(0,fmax))
        isi_coll.append(hist[0])
    return isi_coll,labels_combined_str
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Encoding")
    parser.add_argument("--seed", type=int, default=6, help="Random seed. Default: -1")
    parser.add_argument("--sim_time", type=float, default=1, help="Simulation Time (s)")
    parser.add_argument("--trials_n", type=int, default=100, help="Trials per stimulus")
    parser.add_argument("--noise", type=float, default=100, help="Jitter noise (% of freq)")
    parser.add_argument("--shift", type=float, default=1, help="Random shift (% of freq)")
    parser.add_argument("--batch_size", type=int, default=100, help="Random shift (% of freq)")
    parser.add_argument("--gpu", action='store_true',help='Use GPU if available')
    parser.add_argument("--epochs",type=int,default=100,help="Number of epochs")
    parser.add_argument("--lr",type=float,default=0.1,help="Learning rate Spikes")
    parser.add_argument('--clock_sim',type=float,default=1e-3,help='Simulation clock (s)')
    parser.add_argument('--regen_stimuli',action='store_true',help='Regenerate stimuli')
    parser.add_argument('--nni_opt',action='store_true',help='NNI optimization')
    parser.add_argument('--tqdm_silence',action='store_true',help='Silence tqdm')
    parser.add_argument('--load_optimal',action='store_true',help='Load optimal parameters')
    parser.add_argument('--early_stop',action='store_true',help='Early stop')
    parser.add_argument('--sprinkle',type=float,default=0,help='Sprinkle some random spikes in the dataset')
    parser.add_argument('--neurons_n',type=int,default=1,help='Number of neurons')
    parser.add_argument('--onefreq',action='store_true',help='Use only one frequency')
    parser.add_argument('--fft',action='store_true',help='Use fft instead of isi')

    parser.add_argument('--f1_start',type=float,default=30,help='Start frequency1')
    parser.add_argument('--f1_end',type=float,default=57,help='End frequency1')
    parser.add_argument('--f1_step',type=float,default=3,help='Step frequency1')
    parser.add_argument('--f2_start',type=float,default=30,help='Start frequency2')
    parser.add_argument('--f2_end',type=float,default=57,help='End frequency2')
    parser.add_argument('--f2_step',type=float,default=9,help='Step frequency2')
    args = parser.parse_args()
    parameters = vars(args)  # copy by reference (checked below)
    try:
        os.mkdir('isi_vs_fft')
    except FileExistsError:
        pass
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
        opt_params = json.load(open('isi_vs_fft_multifreq_opt.txt', 'r'))
        print('Loading opt params',opt_params)
        for key, val in opt_params.items():
            parameters[key] = val
            LOG.debug(print(key))
            assert (args.__dict__[key] == parameters[key])
    parameters['n_in'] = 1
    print(parameters)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    freqs1 = np.arange(args.f1_start, args.f1_end, args.f1_step)
    if args.onefreq:
        freqs2 = None
    else:
        freqs2 = np.arange(args.f2_start, args.f2_end, args.f2_step)
    # parameters = {}
    parameters['device'] = 'cuda:0' if (torch.cuda.is_available() & args.gpu) else 'cpu'
    name = gen_name(freqs1,freqs2,args,parameters)
    isi_or_fft = 'fft' if args.fft else 'isi'
    print('Chosen analysis:',isi_or_fft)
    try:
        if args.regen_stimuli:
            raise FileNotFoundError
        torch.load("isi_vs_fft/"+isi_or_fft+'_coll_'+name+'.pt')
        print('Loading'+ isi_or_fft+'_coll_'+name+'.pt')
        coll = torch.load("isi_vs_fft/"+isi_or_fft+'_coll_'+name+'.pt')
        labels = torch.load("isi_vs_fft/"+isi_or_fft+'_labels_'+name+'.pt')
    except FileNotFoundError:
        if args.fft:
            coll,labels = fft_dataset(args,parameters)
        else:
            coll,labels = isi_dataset(args,parameters)
        torch.save(coll,"isi_vs_fft/"+isi_or_fft+'_coll_'+name+'.pt')
        torch.save(labels,"isi_vs_fft/"+isi_or_fft+'_labels_'+name+'.pt')
    coll = torch.tensor(np.vstack(coll)).to(torch.float32)
    coll -= coll.min()
    coll /= coll.max()

    x_tr, x_te, y_tr, y_te = train_test_split(coll,
                                                              labels, test_size=0.1,stratify=labels,random_state=args.seed)
    if args.nni_opt:
        x_tr, x_te, y_tr, y_te = train_test_split(x_tr,
                                                                  y_tr, test_size=0.1,stratify=y_tr,random_state=args.seed)

    dl_tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    dl_te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=True)
    labels_unique = torch.unique(y_tr)
    linear_isi = nn.Linear(x_tr.shape[1], labels_unique.shape[0])

    criterion = nn.CrossEntropyLoss()
    optimizer_isi = torch.optim.Adamax(linear_isi.parameters(), lr=args.lr)
    epochs = trange(int(args.epochs), desc="Training: ", leave=True)
    acc_te = []
    acc_tr = []
    loss_te = []
    loss_tr = []
    for epoch in epochs:
        loss_tr_b = []
        loss_te_b = []
        acc_tr_b = []
        acc_te_b = []
        for x, y in dl_tr:
            optimizer_isi.zero_grad()
            y_pred = linear_isi(x.float())
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer_isi.step()
            loss_tr_b.append(loss.item())
            acc_tr_b.append((y_pred.argmax(dim=1) == y.long()).float().mean().item())
        with torch.no_grad():
            for x, y in dl_te:
                y_pred = linear_isi(x.float())
                loss = criterion(y_pred, y.long())
                loss_te_b.append(loss.item())
                acc_te_b.append((y_pred.argmax(dim=1) == y.long()).float().mean().item())
        acc_tr.append(np.mean(acc_tr_b))
        acc_te.append(np.mean(acc_te_b))
        loss_tr.append(np.mean(loss_tr_b))
        loss_te.append(np.mean(loss_te_b))
        epochs.set_description("Training: (loss=%.4f, acc=%.4f), Testing: (loss=%.4f, acc=%.4f)" % (
            np.mean(loss_tr_b), np.mean(acc_tr_b), np.mean(loss_te_b), np.mean(acc_te_b)))
        if args.nni_opt:
            nni.report_intermediate_result(np.mean(acc_te_b))

    torch.save(linear_isi.state_dict(), 'isi_vs_fft/linear_'+ isi_or_fft + '_' + name+'.pt')

    torch.save(acc_tr, 'isi_vs_fft/acc_tr_'+ isi_or_fft + '_' + name+'.pt')
    print('accuracy:',acc_te)
    torch.save(acc_te, 'isi_vs_fft/acc_te_'+ isi_or_fft + '_' + name+'.pt')
    if args.nni_opt:
        nni.report_final_result(np.mean(acc_te))
