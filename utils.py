from pathlib import Path
import yaml
from cop_class import InstanceBatchManager

class TrainingManager():
    """
    Used to monitor and manage the training.
    The current implementation does not cycle smaller implementations
    faster (we could do 2 episodes of a 2x smaller instance in the same
    nb of steps than the biggest instance of the batch), #TODO: for now
    it 'fills' the now empty threads with unfinished bigger instances.

    FIXME: 
     - 'THREAD' reoccupation when shorter instances are complete.
    TODO: 
     - add performance stopping
     - add sequential action masks where the masks are calculated based
       on the previous mask, and not in 'closed form'
    """

    def __init__(self,cfg,batch:InstanceBatchManager) -> None:
        self.episode = 0
        self.ep_step = 0
        self.best_sol = None
        self.best_sol_eval = None
        self.cfg = cfg
        self.batch = batch
        self.horizon = cfg['horizon']
        self.best_perf = None


    def stop_criterion(self):

        match self.cfg['stop_type']:

            case 'inf':
                return True
            
            case 'ep':
                return self.episode > self.cfg['n_ep']
            


    def get_training_state(self):

        return (
            self.batch.states,
            self.batch.segments,
            self.ep_step,

        )
    
    def new_ep(self):
        return self.ep_step == 0


    def step(self,new_states):

        self.ep_step += self.horizon
        self.batch.states = new_states

        if self.new_ep():
            print(f"EPISODE {self.episode} : {self.best_perf}")


# TODO: check that config is valid
def load_config(path:Path):
    
    with open(path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)


    if not cfg['networks']['dim_embed'] % cfg['networks']['actor']['nhead']:
        raise ValueError('The number of heads needs to be a divisor of the embedding dimension')


    return cfg
