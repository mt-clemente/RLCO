from pathlib import Path
from ppo import PPOTrainer
import argparse
from problems.eternity_cop import Eternity

"""

TODO:

 - For now only supporting tensor representations
 - Cutsom performance indicators
 - Use partial solutions for transf?? Aka manage sequence length, is it useful? --> Not for now

"""



parser = argparse.ArgumentParser()
parser.add_argument('--files', dest='files',type=str,default='instances')

args = parser.parse_args()

problem = Eternity

trainer = PPOTrainer(
    Path("config.yml"),
    Eternity
)

trainer.train(
    args.files
)