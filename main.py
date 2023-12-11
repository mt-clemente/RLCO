from pathlib import Path

import torch
import wandb
from ppo_solver import RLCOSolver
import argparse
from problems.eternity_cop import Eternity
from problems.tsp import TSP

"""
TODO:

    - GTRXL
    - Check for the impact of the transformer sequence going over the end --> longer sequence might give a better vision
    |
    --> gtrxl could learn from solving the same problem over and over

    - Add sequence length choice

    
 - Buf state storage unit
 - For now only supporting tensor representations
 - Cutsom performance indicators
 - Use partial solutions for transf?? Aka manage sequence length, is it useful? --> Not for now

"""
# torch.cuda.is_available = lambda : False
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg',type=str,default='config.yml')
parser.add_argument('--files', dest='files',type=str,default='instances/eternity_B.txt')

args = parser.parse_args()

problem = TSP


trainer = RLCOSolver(
    config_path=args.cfg,
    instances_path=Path(args.files),
    problem=problem,
)

wandb.init(
    project='EteRLnity',
    entity='mateo-clemente',
    group='NewTests',
    config=trainer.cfg
)

trainer.train()

wandb.finish()
