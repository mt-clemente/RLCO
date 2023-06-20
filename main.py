from pathlib import Path

import torch
from solver import RLCOSolver
import argparse
from problems.eternity_cop import Eternity

"""
Cool because is anytime


FIXME:
 - categorical vocab size ----> For now only works with categorical ---> No embedding needed for non categorical ----> ??
 - Fix causal mask (too slow version)


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
torch.cuda.is_available = lambda : False


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg',type=str,default='config.yml')
parser.add_argument('--files', dest='files',type=str,default='instances')

args = parser.parse_args()

problem = Eternity

trainer = RLCOSolver(
    config_path=args.cfg,
    instances_path=Path(args.files),
    problem=Eternity,
)

trainer.train()