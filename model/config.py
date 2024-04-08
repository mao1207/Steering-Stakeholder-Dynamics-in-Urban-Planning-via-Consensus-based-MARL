# config.py
import argparse
import torch
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment and Training Settings.")
    
    # Experiment-specific settings
    parser.add_argument('--experiment_mode', type=str, default="random", choices=['actor-critic', 'greedy', 'random'], help="Choose the method: actor-critic or greedy.")
    parser.add_argument('--if_only_top_down', type=bool, default=False)

    # The following is the training setup for GNN-DRL
    parser.add_argument('--model_mode', type=str, default='GCN')
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--n_hid', type=int, default=512)
    parser.add_argument('--n_inp', type=int, default=19)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--max_lr', type=float, default=5e-4)

    return parser.parse_args()

args = parse_arguments()

# Device Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Other global variables
max_distance = 1250

script_dir = os.path.dirname(os.path.abspath(__file__))

scale_num = 0.99

locate_one_hot = torch.rand((3, 5)).to(device)