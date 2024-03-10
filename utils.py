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
from sklearn.model_selection import train_test_split
import nni
import logging
import os
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import preprocessing


import torch.nn.functional as F


class FocalLoss(nn.Module):
    "Focal loss implemented using F.cross_entropy"
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction


    def forward(self, inp: torch.Tensor, targ: torch.Tensor):
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

def gen_name(freqs1,freqs2,args,parameters):
    freq1_desc = 'f1_' + str(freqs1[0]) + '_' + str(freqs1[1] - freqs1[0]) + '_' + str(freqs1[-1])
    if freqs2 is not None:
        freq2_desc = 'f2_' + str(freqs2[0]) + '_' + str(freqs2[1] - freqs2[0]) + '_' + str(freqs2[-1])
    else:
        freq2_desc = ''
    print('freqs2', freqs2)
    trials_desc = str(args.trials_n)
    noise_desc = str(args.noise)
    shift_desc = str(args.shift)
    sprinkle_desc = str(args.sprinkle)
    seed_desc = str(args.seed)
    sim_time_desc = str(parameters['sim_time'])

    name = freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + '-' + sprinkle_desc + '_' + sim_time_desc + '-' + seed_desc
    return name
def define_one_freq(freqs1,trials_n,noise,shift,parameters,in_spikes_list=[],labels=[],sprinkle_mag=0):
    freqs_list = []
    # print('in_spikes_list',in_spikes_list)
    print('defining one freq')
    for freq1_idx,freq1 in enumerate(freqs1):
        # print('freq1_idx',freq1_idx)
        for trial in range(trials_n):
            # print('trial',trial)
            freqs_sparse = torch.arange(0,parameters['sim_time']/parameters['clock_sim'],np.ceil(1/parameters['clock_sim']/freq1))
            noise_contr = torch.randn_like(freqs_sparse)*noise/(parameters['clock_sim']*freq1)*parameters['clock_sim']
            shift_contr = torch.ones_like(freqs_sparse)*torch.rand(1)/(parameters['clock_sim']*freq1)*parameters['clock_sim']*shift
            n_to_sprinkle = (torch.rand(1))*parameters['sim_time']/parameters['clock_sim']*sprinkle_mag
            sprinkle = torch.randint(0,int(parameters['sim_time']/parameters['clock_sim']),[int(n_to_sprinkle)])
            freqs_sparse += noise_contr + shift_contr
            freqs_sparse = torch.cat((freqs_sparse,sprinkle))
            in_spikes = torch.sparse_coo_tensor(freqs_sparse.unsqueeze(0),torch.ones_like(freqs_sparse),size=([int(np.ceil(parameters['sim_time']/parameters['clock_sim']))])).to_dense()
            freqs_list.append(freqs_sparse)
            # print(in_spikes.shape)
            in_spikes_list.append(in_spikes)
            # print('len in_spikes_list',len(in_spikes_list))
            labels.append(freq1_idx)
    in_spikes_s = torch.stack(in_spikes_list,dim=0).T
    in_spikes_s = in_spikes_s.unsqueeze(2)*torch.ones((1,1,parameters['neurons_n']))
    # print('single spike shape',in_spikes_s.shape)
    in_spikes_s[in_spikes_s>0] = 1
    in_spikes_s = in_spikes_s.to(parameters['device'])
    # aaa = torch.where(in_spikes_s==1)
    # plt.scatter(aaa[0],aaa[1],s=0.1)
    # print(freqs_list)
    return in_spikes_s,freqs_list,labels
def define_two_freqs(freqs1,freqs2, trials_n, noise,shift,parameters,sprinkle_mag=0):
    in_spikes_s,freqs_list,labels = define_one_freq(freqs1=freqs1,trials_n=trials_n,noise=noise,shift=shift,parameters=parameters,in_spikes_list=[],labels=[],sprinkle_mag=sprinkle_mag)
    in_spikes_list = []
    labels_combined = []
    print('defining two freqs')
    # print(trial)
    for fr2,freq2 in enumerate(freqs2):
        freqs_sparse = torch.arange(0,parameters['sim_time']/parameters['clock_sim'],int(1/parameters['clock_sim']/freq2))
        freqs_list_tmp = []
        for fr1,freq1 in enumerate(freqs1):
            selected_trials = np.where(np.array(labels)==fr1)[0]
            # print(selected_trials)
            for trial in selected_trials:
                noise_contr = torch.randn_like(freqs_sparse) * noise / (parameters['clock_sim'] * freq2) * parameters[
                    'clock_sim']
                shift_contr = torch.ones_like(freqs_sparse) * torch.rand(1) / (parameters['clock_sim'] * freq2) * \
                              parameters['clock_sim'] * shift
                # print(noise_contr)
                # print(shift_contr)
                freqs_sparse_tmp = freqs_sparse + noise_contr + shift_contr
                n_to_sprinkle = torch.rand(1) * parameters['sim_time'] / parameters['clock_sim'] * sprinkle_mag
                sprinkle = torch.randint(0, int(parameters['sim_time'] / parameters['clock_sim']), [int(n_to_sprinkle)])
                freqs_sparse = torch.cat((freqs_sparse, sprinkle))
                # print('len freq_list',len(freqs_list))
                # print('trial',trial)
                freqs_list_tmp.append(torch.cat((freqs_list[trial],freqs_sparse_tmp)))

                # freqs_list_tmp.append(freqs_list[trial])
                in_spikes = torch.sparse_coo_tensor(freqs_list_tmp[trial].unsqueeze(0),torch.ones_like(freqs_list_tmp[trial]),size=([int(parameters['sim_time']/parameters['clock_sim'])])).to_dense()
                in_spikes_list.append(in_spikes)
                labels_combined.append([np.where(freqs1==freq1)[0][0],np.where(freqs1==freq2)[0][0]])
    in_spikes_m = torch.stack(in_spikes_list,dim=0).T
    in_spikes_m = in_spikes_m.unsqueeze(2)*torch.ones((1,1,parameters['neurons_n']))
    in_spikes_m[in_spikes_m>0] = 1
    in_spikes_m = in_spikes_m.to(parameters['device'])
    # aaa = torch.where(in_spikes_m==1)
    # plt.scatter(aaa[0],aaa[1],s=0.1)
    return in_spikes_m,labels_combined

def tensor_t(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))
def sparse_tensor_flatten(x, start_dim=1, end_dim=2):
    indices = x.coalesce().indices()
    uniques = torch.unique(indices[end_dim])
    coll = []
    for unique in uniques:
        coll.append(sparse_tensor_stride(x,[None,None,unique]).coalesce())
    return torch.stack(coll, dim=start_dim)

def iterative_train_test_split_sparse(sparse_data, labels, test_size=0.1):
    sparse_data = sparse_data.coalesce()

    time_idx = sparse_data.indices()[1]
    trial_idx = sparse_data.indices()[0]
    neuron_idx = sparse_data.indices()[2]

    probability_dist_labels = torch.zeros((labels.shape[0], torch.unique(labels).shape[0]))
    for i in range(labels.shape[0]):
        probability_dist_labels[i, labels[i].to(torch.int)] += 1
    X_train_idx, y_train, X_test_idx, y_test = iterative_train_test_split(
        torch.tensor([i for i in range(labels.shape[0])]).unsqueeze(1),
        probability_dist_labels, test_size=test_size)
    X_train_idx = X_train_idx[:, 0]
    selection_criteria = torch.zeros_like(trial_idx, dtype=torch.bool)
    for X_train_idx_s in range(X_train_idx.shape[0]):
        selection_criteria = selection_criteria | (trial_idx == X_train_idx[X_train_idx_s])

    indices = torch.stack([trial_idx[selection_criteria],time_idx[selection_criteria], neuron_idx[selection_criteria]])

    size = [X_train_idx.shape[0],sparse_data.shape[1], sparse_data.shape[2]]
    X_train = torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(tensor_t(trial_idx[selection_criteria])), size=size).coalesce()


    X_test_idx = X_test_idx[:, 0]
    selection_criteria = torch.zeros_like(trial_idx, dtype=torch.bool)
    for X_test_idx_s in range(X_test_idx.shape[0]):
        selection_criteria = selection_criteria | (trial_idx == X_test_idx[X_test_idx_s])
    indices = torch.stack([trial_idx[selection_criteria], time_idx[selection_criteria], neuron_idx[selection_criteria]])
    size = [X_test_idx.shape[0],sparse_data.shape[1], sparse_data.shape[2]]
    X_test = torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(tensor_t(trial_idx[selection_criteria])), size=size).coalesce()
    print(X_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, y_test

def select_trials(trial_idx, indices, X_idx, sparse_data,trial_pos):
    '''
    :param trial_idx: trial index of the sparse data
    :param time_idx: time index of the sparse data
    :param neuron_idx: neuron index of the sparse data
    :param X_idx: indices of the trials to be selected
    :param sparse_data: sparse data
    :return: sparse data with only the selected trials
    Slices the sparse data to only include the trials specified in X_idx
    '''
    selection_criteria = torch.zeros_like(trial_idx, dtype=torch.bool)
    for X_idx_i,X_idx_s in enumerate(X_idx):
        selection_criteria = selection_criteria | (trial_idx == X_idx_s)
    for X_idx_i, X_idx_s in enumerate(X_idx):
        trial_idx[trial_idx == X_idx_s] = X_idx_i
    # trial_idx_updated = torch.arange(trial_idx[selection_criteria].shape[0])
    indices_to_stack = [trial_idx[selection_criteria]]

    for indice in indices:
        indices_to_stack.append(indice[selection_criteria])
    size = [X_idx.shape[0]] 
    for size_sweep in range(len(sparse_data.shape)):
        if size_sweep !=trial_pos:
            size.append(sparse_data.shape[size_sweep])
    indices = torch.stack(indices_to_stack)
    #indices = torch.stack([trial_idx[selection_criteria], time_idx[selection_criteria], neuron_idx[selection_criteria]])

    #size = [X_idx.shape[0], sparse_data.shape[1], sparse_data.shape[2]]
    X = torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(tensor_t(trial_idx[selection_criteria])),
                                      size=size)
    return X

def train_test_split_sparse(sparse_data, labels, test_size=0.1, random_state=None,stratify=None,shuffle=None,trial_pos=0):
    '''
    :param sparse_data: sparse tensor
    :param labels: labels
    :param test_size: test size
    :return: X_train, y_train, X_test, y_test
    Wrapper for train_test_split from sklearn in order to split sparse tensors
    The idea is that we first split the labels into train and test and then
    select the corresponding trials from the sparse tensor
    '''
    sparse_data = sparse_data.coalesce()
    indices = []
    for indice in range(len(sparse_data.indices())):
        if indice == trial_pos:
            trial_idx = sparse_data.indices()[indice]
        else:
            indices.append(sparse_data.indices()[indice])
    # sparse_data_tmp = sparse_tensor_stride(sparse_data, [None,None,0]).coalesce()
    # plt.scatter(sparse_data_tmp.indices()[1],sparse_data_tmp.indices()[0])
    # plt.show()

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        torch.tensor([i for i in range(labels.shape[0])]).unsqueeze(1),
        labels, test_size=test_size, random_state=random_state,stratify=stratify,shuffle=shuffle)

    X_train_idx = X_train_idx[:, 0]
    X_train = select_trials(trial_idx, indices, X_train_idx, sparse_data,trial_pos=trial_pos)

    X_test_idx = X_test_idx[:, 0]
    X_test = select_trials(trial_idx, indices, X_test_idx, sparse_data,trial_pos=trial_pos)
    return X_train, X_test, y_train,y_test

def sparse_tensor_stride(sparse_tensor, strides):
    '''
    :param sparse_tensor: sparse tensor
    :param strides: strides
    :return: sparse tensor with strides
    This function is meant to implement the indexing of sparse tensors
    You give it a sparse tensor and a list of strides and it returns the sparse tensor with the strides
    If you want a slice you can use None
    ex. sparse_tensor_stride(sparse_tensor, [5,4,2]) will return a sparse tensor with strides 5,4,2
    ex. sparse_tensor_stride(sparse_tensor, [None,None,0]) will return a sparse tensor with strides :,:,0
    '''
    indices = sparse_tensor.coalesce().indices()
    selection_criteria = torch.ones_like(indices[0], dtype=torch.bool)
    shape = []
    for stride_idx,stride in enumerate(strides):
        if type(stride) == int:
            strides[stride_idx] = torch.tensor([stride])
            shape.append(1)
        elif type(stride) == list:
            strides[stride_idx] = torch.tensor(stride)
            shape.append(len(stride))
        elif type(stride) == torch.Tensor:
            strides[stride_idx] = torch.tensor([stride])
            shape.append(stride.shape[0])
        elif stride == None:
            strides[stride_idx] = torch.unique(indices[stride_idx])
            shape.append(sparse_tensor.shape[stride_idx])
        else:
            raise TypeError('Stride must be int, list,torch.Tensor or None. Type was: '+str(type(stride))+'.')
    for stride_idx,stride in enumerate(strides):
        selection_criteria_local = torch.zeros_like(indices[0], dtype=torch.bool)
        for index in range(stride.shape[0]):
            selection_criteria_local = selection_criteria | (indices[stride_idx] == index)

        selection_criteria = selection_criteria & selection_criteria_local
    indices_new = []

    for ind_idx in range(len(indices)):

        indices_new.append(indices[ind_idx][selection_criteria])
    indices = torch.stack(indices_new)
    return torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(indices[0]), size=shape)
def accuracy_multi(inp, targ, thresh=0.5):
    "Compute accuracy when `inp` and `targ` are the same size."

    return ((inp>thresh)==targ.bool()).float().mean()

def import_dataloaders(freqs1,freqs2,args,parameters,sparse=False):
    freq1_desc = 'f1_' + str(freqs1[0]) + '_' + str(freqs1[1] - freqs1[0]) + '_' + str(freqs1[-1])
    freq2_desc = 'f2_' + str(freqs2[0]) + '_' + str(freqs2[1] - freqs2[0]) + '_' + str(freqs2[-1])
    trials_desc = str(args.trials_n)
    noise_desc = str(args.noise)
    shift_desc = str(args.shift)
    sprinkle_desc = str(args.sprinkle)
    seed_desc = str(args.seed)
    sim_time_desc = str(parameters['sim_time'])

    name = freq1_desc + '-' + freq2_desc + '-' + trials_desc + '-' + noise_desc + '-' + shift_desc + '-' + sprinkle_desc + '_' + sim_time_desc + '-' + seed_desc
    return name
def import_dataloaders(freqs1,freqs2,args,parameters,sparse=False,return_datasets=False):

    try:
        os.mkdir('synthetic_data')
    except FileExistsError:
        pass
    name = gen_name(freqs1,freqs2,args,parameters)
    file_spikes = 'in_spikes_m-' + name + '.pt'
    file_labels_combined = 'labels_combined-' + name + '.pt'
    try:
        if args.regen_stimuli:
            raise FileNotFoundError
        in_spikes_m = torch.load(os.path.join('synthetic_data', file_spikes), map_location=parameters['device'])
        labels_combined = torch.load(os.path.join('synthetic_data', file_labels_combined),
                                     map_location=parameters['device'])
    except FileNotFoundError:
        print('Generating new stimuli')
        if freqs2 is None:
            in_spikes_m,_, labels_combined = define_one_freq(freqs1=freqs1,trials_n=args.trials_n,noise=args.noise,shift=args.shift,parameters=parameters,in_spikes_list=[],labels=[],sprinkle_mag=args.sprinkle)
        else:
            in_spikes_m, labels_combined = define_two_freqs(freqs1=freqs1,freqs2=freqs2,trials_n=args.trials_n,noise=args.noise,shift=args.shift,parameters=parameters,sprinkle_mag=args.sprinkle)
        labels_combined = torch.tensor(labels_combined)
        torch.save(in_spikes_m, os.path.join('synthetic_data', file_spikes))
        torch.save(labels_combined, os.path.join('synthetic_data', file_labels_combined))
    # print('in_spikes_m.shape', in_spikes_m.shape)

    # in_spikes_m_noneuron = in_spikes_m.to_dense()[:,:,0]
    # aaa = torch.where(in_spikes_m_noneuron)
    # plt.scatter(aaa[0], aaa[1], s=0.1)
    # plt.show()
    parameters['trials_per_stimulus'] = in_spikes_m.shape[1]
    batch_size = int(args.batch_size)

    if freqs2 is not None:
        labels_combined_str = [str([label.item() for label in labels]) for labels in labels_combined]
    else:
        labels_combined_str = [str(label.item()) for label in labels_combined]
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels_combined_str))
    labels_combined_idx = le.transform(labels_combined_str)
    labels_combined_idx = torch.tensor(labels_combined_idx)
    # labels_combined_idx = torch.tensor(np.unique(labels_combined_str, return_inverse=True)[1])
    X_train, X_test, y_train, y_test = train_test_split(in_spikes_m.permute(1,0,2).to('cpu'),
                                                                  labels_combined_idx, test_size=0.1,
                                                        random_state=args.seed,shuffle=True,stratify=labels_combined_idx)
    eee = torch.unique(y_train,return_counts=True)
    uuu = torch.unique(y_test,return_counts=True)
    # print(eee)
    # print(uuu)
    # in_spikes_m_noneuron = X_train.to_dense()[:,:,0].T
    # aaa = torch.where(in_spikes_m_noneuron)
    # plt.figure()
    # plt.scatter(aaa[0], aaa[1], s=0.1)
    # in_spikes_m_noneuron = X_test.to_dense()[:, :, 0].T
    # aaa = torch.where(in_spikes_m_noneuron)
    # plt.figure()
    # plt.scatter(aaa[0], aaa[1], s=0.1)
    # plt.show()
    if args.nni_opt:
        X_train, X_test, y_train, y_test = train_test_split(X_train.to('cpu'),
                                                            y_train, test_size=0.1,random_state=args.seed,stratify=y_train)
    if sparse == True:
        X_train = X_train.to_sparse()
        X_test = X_test.to_sparse()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if return_datasets:
        return in_spikes_m.permute(1,0,2),labels_combined_idx
    return train_dataloader, test_dataloader, labels_combined_str