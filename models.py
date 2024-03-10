import torch
import torch.nn as nn
from collections import namedtuple

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply

class OSC_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'I','count_refr','spk'])

    def __init__(self, parameters):
        super(OSC_neuron, self).__init__()
        self.C = parameters['C_Osc']
        self.thr = parameters['v_Osc_threshold']
        self.state = None
        # self.bs = parameters['trials_per_stimulus']
        self.n = parameters['neurons_n']
        self.tau_syn = parameters['tau_Osc_Ie']
        self.tau_neu = parameters['tau_Osc']
        self.gain_syn = parameters['TDE_to_Osc_current']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_osc']
        self.device = parameters['device']
        self.I_minimum_osc = parameters['I_minimum_osc']
        self.I_step_osc = parameters['I_step_osc']
        self.refrac_Osc = parameters['refrac_Osc']
        self.steady_current = torch.tensor([self.I_minimum_osc + self.I_step_osc * i for i in range(self.n)]).to(self.device)
        self.refr = self.refrac_Osc/self.dt

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones((self.bs,self.n), device=input.device),
                                          I=torch.zeros((self.bs,self.n), device=input.device),
                                          count_refr = torch.zeros((self.bs,self.n), device=input.device),
                                            spk=torch.zeros((self.bs,self.n), device=input.device)
                                          )
        V = self.state.V
        I = self.state.I
        count_refr = self.state.count_refr
        I += -self.dt * I / self.tau_syn + self.gain_syn * input
        V += self.dt * (-V / self.tau_neu + (I + self.steady_current) / self.C)
        spk = activation(V - self.thr)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.Vr
        self.state = self.NeuronState(V=V, I=I,count_refr = count_refr,spk=spk)
        return spk

class TDE(nn.Module):
    NeuronState = namedtuple('NeuronState', ['v', 'i_trg', 'i_fac','count_refr','spk'])

    def __init__(self, parameters):
        super(TDE, self).__init__()
        self.C = parameters['C_TDE']
        self.thr = parameters['v_TDE_threshold']
        self.state = None
        # self.bs = parameters['trials_per_stimulus']
        self.n = parameters['neurons_n']
        self.tau_trg = parameters['tau_trg_TDE']
        self.tau_fac = parameters['tau_fac_TDE']
        self.gain_fac = parameters['gain_fac_TDE']
        self.gain_trg = parameters['gain_trg_TDE']
        self.beta = parameters['tau_TDE']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_TDE']
        self.refr = parameters['refrac_TDE']/self.dt
        self.device = parameters['device']
        # self.refr.to(self.device)

    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input_trg, input_fac):
        if self.state is None:
            self.state = self.NeuronState(v=torch.zeros(input_trg.shape, device=input_trg.device),
                                          i_trg=torch.zeros(input_trg.shape, device=input_trg.device),
                                          i_fac=torch.zeros(input_trg.shape, device=input_trg.device),
                                          count_refr = torch.zeros((self.bs,self.n), device=input_trg.device),
                                          spk = torch.zeros(input_trg.shape, device=input_trg.device)
                                          )

        assert input_trg.shape == input_fac.shape, "TRG and FAC should have same dimensions, while: shape trg: " + str(input_trg.shape) + ", shape fac: " + str(input_fac.shape)
        v = self.state.v
        i_trg_prev = self.state.i_trg
        i_fac_prev = self.state.i_fac
        count_refr = self.state.count_refr
        i_fac = i_fac_prev -self.dt / self.tau_fac * i_fac_prev + self.gain_fac * input_fac
        i_trg = i_trg_prev -self.dt / self.tau_trg * i_trg_prev + self.gain_trg * input_trg * i_fac * 10000
        v += self.dt * (- v / self.beta + i_trg / self.C)
        spk = activation(v - self.thr)
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        v = (1 - spk) * v * (count_refr <= 0) + spk * self.Vr
        self.state = self.NeuronState(v=v, i_trg=i_trg, i_fac=i_fac,count_refr= count_refr,spk=spk)
        return spk

class sPLL(nn.Module):
    def __init__(self, parameters):
        super(sPLL, self).__init__()
        self.device = parameters['device']
        self.TDE = TDE(parameters).to(self.device)
        self.OSC = OSC_neuron(parameters).to(self.device)
        self.n_out = parameters['neurons_n']
        self.n_in = 1
        try:
            self.tmp_current_list = parameters['current_list']
        except:
            self.tmp_current_list = torch.linspace(parameters['I_minimum_osc'],parameters['I_minimum_osc'] + parameters['I_step_osc']*self.n_out,self.n_out).to(self.device)
        # print('TDE_gain',parameters['gain_trg_TDE'])
        # print('tmp_current_list',self.tmp_current_list)
    def initialize_state(self,parameters):
        self.OSC.initialize_state(parameters)
        self.TDE.initialize_state(parameters)
        self.bs = parameters['trials_per_stimulus']
        self.spikes_TDE_prev = torch.zeros((self.bs,self.n_out), device=self.device)
        self.current_list = (self.tmp_current_list.to(self.device)*torch.ones_like(self.spikes_TDE_prev))
        self.OSC.steady_current = self.current_list
    def forward(self, input):
        # input_expanded = input*torch.ones([input.shape[0],self.n_out]).to(self.device).T
        spikes_OSC = self.OSC(self.spikes_TDE_prev)
        spikes_TDE = self.TDE(spikes_OSC,input)
        self.spikes_TDE_prev = spikes_TDE.clone().detach()
        return spikes_TDE, spikes_OSC


class LIF_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'Isyn','Irec','count_refr','spk'])

    def __init__(self, parameters,train=False,recurrent=False):
        super(LIF_neuron, self).__init__()
        self.C = parameters['C_Osc']
        self.thr = parameters['v_Osc_threshold']
        self.state = None
        self.bs = parameters['trials_per_stimulus']
        self.n = parameters['neurons_n']
        self.tau_syn = parameters['tau_Osc_Ie']
        self.tau_syn_rec = parameters['tau_Osc_Irec']
        self.tau_neu = parameters['tau_Osc']
        self.gain_syn = parameters['TDE_to_Osc_current']
        self.gain_syn_rec = parameters['gain_syn_rec']
        self.dt = parameters['clock_sim']
        self.Vr = parameters['reset_osc']
        self.device = parameters['device']
        self.I_minimum_osc = parameters['I_minimum_osc']
        self.I_step_osc = parameters['I_step_osc']
        self.refrac_Osc = parameters['refrac_Osc']
        self.steady_current = torch.tensor([self.I_minimum_osc + self.I_step_osc * i for i in range(self.n)]).to(self.device)
        self.refr = self.refrac_Osc/self.dt
        self.n_in = parameters['n_in']
        if train == True:
            self.h1 = nn.Parameter(torch.empty(self.n_in, self.n), requires_grad=True)
            torch.nn.init.normal_(self.h1, mean=1, std=parameters['weight_std']*0.5/torch.sqrt(torch.tensor(self.n_in).float()))
            print('h1',self.h1)
            if recurrent:
                self.r1 = nn.Parameter(torch.empty(self.n, self.n), requires_grad=True)
                torch.nn.init.normal_(self.r1, mean=0, std=parameters['weight_std']*0.5/torch.sqrt(torch.tensor(self.n_in).float()))
            else:
                self.r1 = torch.zeros(self.n, self.n).to(self.device)
        else:
            self.h1 = torch.ones(self.n_in, self.n).to(self.device)
            if recurrent:
                self.r1 = torch.ones(self.n, self.n).to(self.device)
            else:
                self.r1 = torch.zeros(self.n, self.n).to(self.device)


    def initialize_state(self,parameters):
        self.state = None
        self.bs = parameters['trials_per_stimulus']
    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones((self.bs,self.n), device=input.device),
                                          Isyn=torch.zeros((self.bs,self.n), device=input.device),
                                          Irec=torch.zeros((self.bs,self.n), device=input.device),
                                          spk=torch.zeros((self.bs, self.n), device=input.device),
                                          count_refr = torch.zeros((self.bs,self.n), device=input.device)
                                          )
        Vold = self.state.V
        Isyn_old = self.state.Isyn
        Irec_old = self.state.Irec
        spk = self.state.spk
        count_refr = self.state.count_refr
        I_syn = Isyn_old -self.dt * Isyn_old / self.tau_syn + self.gain_syn * torch.mm(input,self.h1)
        I_rec = Irec_old - self.dt * Irec_old / self.tau_syn_rec + self.gain_syn_rec * torch.mm(spk,self.r1)
        # print('Isyn',I_syn)
        # print('Irec',I_rec)
        V = Vold + self.dt * (-Vold / self.tau_neu + (I_syn + I_rec + self.steady_current) / self.C)
        V = torch.clamp(V, min=0)
        # print('V',V)
        spk = activation(V - self.thr)
        eee = torch.where(spk)
        # print('V',V)
        # print('spk',eee)
        # if len(eee[0])>0:
        #     print('spike')
        count_refr = self.refr*(spk) + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.Vr
        self.state = self.NeuronState(V=V, Isyn=I_syn,Irec=I_rec,count_refr = count_refr,spk=spk)
        return spk
