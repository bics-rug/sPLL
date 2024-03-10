# NNi uses an experiment to manage the HPO process.
# The experiment config defines how to train the models and how to explore the search space.

# To retrieve the experiment results:
# nnictl view [experiment_name] --port=[port] --experiment_dir=[EXPERIMENT_DIR]
#
# For example:
# nnictl view zk29xumi --port=8080 --experiment_dir=/home/p308783/nni-experiments/eb_fall_detection/


import argparse
import json
import os
import time

from nni.experiment import ExperimentConfig, AlgorithmConfig, LocalConfig, Experiment, RemoteConfig, RemoteMachineConfig

# search_space = {
#     'lr': {'_type': 'loguniform',
#            '_value': [0.01,1]},
#     'n_layers' : {'_type': 'choice',
#                     '_value': [1,2,3]},
#     'n_neurons_l0': {'_type': 'choice',
#                      '_value': [50, 100, 150]},
#     'n_neurons_l1': {'_type': 'choice',
#                      '_value': [50, 100, 150]},
#     'n_neurons_l2': {'_type': 'choice',
#                      '_value': [50, 100, 150]},
#     'thr_acc': {'_type': 'uniform',
#                 '_value': [0.5, 1]},
#     'gain_syn': {'_type': 'loguniform',
#                 '_value': [100,10000]},
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument('--exp_name',
                        type=str,
                        default='LIF_multifreq',
                        help='Name for the starting experiment.')

    # Maximum number of trials
    parser.add_argument('--exp_trials',
                        type=int,
                        default=100,
                        help='Number of trials for the starting experiment.')

    # Maximum time
    parser.add_argument('--exp_time',
                        type=str,
                        default='2d',
                        help='Maximum duration of the starting experiment.')

    # How many (if any) GPUs to use
    parser.add_argument('--exp_gpu_number',
                        type=int,
                        default=1,
                        help='How many GPUs to use for the starting experiment.')

    # Which GPU to use
    parser.add_argument('--exp_gpu_sel',
                        type=int,
                        default=0,
                        help='GPU index to be used for the experiment.')

    # How many trials at the same time
    parser.add_argument('--exp_concurrency',
                        type=int,
                        default=1,
                        help='Concurrency for the starting experiment.')


    # Which port to use
    parser.add_argument('--port',
                        type=int,
                        default=8080,
                        help='Port number for the starting experiment.')

    # How many epochs per trial
    parser.add_argument('--n_epochs',
                        type=int,
                        default=200,
                        help='Port number for the starting experiment.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1000,
                        help='Experiment batch size.')


    parser.add_argument('--generate_splits',
                        type=bool,
                        default=False,
                        help='If true, generate train test splits first')

    parser.add_argument('--n_splits',
                        type=int,
                        default=None,
                        help='N train/validation splits to generate')

    parser.add_argument('--splits_seed',
                        type=int,
                        default=6,
                        help='Value seed used to generate input splits')

    parser.add_argument('--platform',
                        type=str,
                        default='local',
                        choices=['local', 'remote'],
                        help='platform where to run experiment')

    parser.add_argument('--tuner',
                        type=str,
                        default='Anneal',
                        choices=['GridSearch', 'Anneal'],
                        help='Tuner algorithm')

    parser.add_argument('--location',
                        type=str,
                        default=None,
                        help=' If not None, this is used to specify the input folder from which data is loaded.')
    parser.add_argument('--early_stop',
                        type=int,
                        default=10,
                        help='Number of epochs to wait before stopping training if validation error does not improve')
    parser.add_argument('--gpu',action='store_true',help='Tell if to use gpu')
    parser.add_argument('--args', nargs='*', help='Command line arguments for the experiment')

    args = parser.parse_args()
    search_space = json.load(open(args.exp_name+'_search.txt', 'r'))
    print(search_space)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = args.exp_name
    if args.gpu:
        is_gpu = ' --gpu'
    else:
        is_gpu = ''
    # Generate train/validation splits
    if args.platform == 'local':  # run locally
        training_service = LocalConfig(trial_gpu_number=args.exp_gpu_number,
                                       use_active_gpu=args.gpu)
        if args.gpu == False:
            training_service = LocalConfig()
            args.exp_gpu_sel = 0
    else:
        training_service = RemoteConfig()  # configure remote training service
        training_service.platform = 'remote'
        machine_gpu = RemoteMachineConfig()
        machine_gpu.host = 'bics.gpu-a4000'
        machine_gpu.user = 'p301974'
        machine_gpu.ssh_key_file = '~/.ssh/id_ed25519'
        machine_gpu.use_active_gpu = args.gpu
        training_service.machine_list = [machine_gpu]

    other_args = ''
    print(args.args)
    if args.args is not None:
        other_args_list = str(args.args[0]).split(',')
        print(other_args_list)
        for arg in other_args_list:
            other_args += ' ' + arg
    print(other_args)
    config = ExperimentConfig(
        experiment_name=args.exp_name,
        experiment_working_directory="~/nni-experiments/{}".format(
            args.exp_name),
    	trial_command=f"python3 {args.exp_name}.py --nni_opt " + is_gpu + other_args,
        trial_code_directory="./",
        search_space=search_space,
        tuner=AlgorithmConfig(name=args.tuner,  # "Anneal",
                              class_args={"optimize_mode": "maximize"}),
        assessor=AlgorithmConfig(name="Medianstop",
                                 # early stopping: Stop if the hyperparameter set performs worse than median at any step.
                                 class_args=({'optimize_mode': 'maximize',
                                              'start_step': args.early_stop})),
        tuner_gpu_indices=args.exp_gpu_sel,
        max_trial_number=args.exp_trials,
        max_experiment_duration=args.exp_time,
        trial_concurrency=args.exp_concurrency,
        training_service=training_service
    )

    experiment = Experiment(config)

    experiment.run(args.port)

    # Stop through input
    input('Press any key to stop the experiment.')

    # Stop at the end
    experiment.stop()
