import copy
from pathlib import Path
import torch
import yaml
from cop_class import COProblem

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

    def __init__(self,cfg,instances_path:Path,problem:COProblem,device) -> None:

        self.device = device
        self.pb = problem
        self.cfg

        # Batch mgt
        instances = []
        for file in instances_path.iterdir():
            inst = self.pb.load_instance(file)
            instances.append(inst)

        self.instances = instances

        if isinstance(self.states[0],torch.Tensor) and isinstance(self.segments[0],torch.Tensor):
            self.states = torch.vstack([i.state for i in self.instances])
            self.segments = torch.vstack([i.segments for i in self.instances])

        self.sizes= torch.tensor([i.size for i in self.instances])
        self.init_states = copy.deepcopy(self.states)


        self.base_masks, self.src_key_padding_masks = self.init_masks()
        self.masks = self.base_masks.clone()

        self.num_instances = len(instances)
        self.max_inst_size = max([i.size for i in self.instances])
        self.instance_lengths = torch.tensor([len(x) for x in self.instances])

        self.reset()

        self.cfg = self.cfg
        self.episode = torch.zeros_like(self.sizes)
        self.curr_step = 0
        self.horizon = self.cfg['horizon']


    def reset(self):

        # Training mgt
        self.best_sol = None
        self.best_sol_eval = None
        self.best_perf = None

        
    def stop_criterion(self):

        match self.cfg['stop_type']:

            case 'inf':
                return True
            
            case 'ep':
                return self.episode > self.cfg['n_ep']
            


    def get_training_state(self):

        return (
            self.states,
            self.segments,
            self.get_ep_step(),
        )
    


    def step(self,actions):

        self.curr_step += 1
        self.masks[actions] = False

        # New episode
        finals = self.curr_step == self.instance_lengths
        self.states[finals] = self.init_states
        self.masks[finals] = self.base_masks

        # FIXME:FIXME:FIXME:FIXME:FIXME:FIXME:FIXME:
        if self.curr_step >= self.max_inst_size: #FIXME: +- 1?
            print(f"EPISODE {self.episode} : {self.best_perf}" )#FIXME: +- Custom perf?
            self.episode += 1
            self.reset()

    def finalize_rollout(self,states):
        self.states = states



    def init_masks(self):
        
        # pad to get all max length sequences
        src_key_padding_mask = torch.arange(self.max_inst_size, device=self.device) > self.instance_lengths
        base_tgt_key_padding_mask = src_key_padding_mask.clone()


        return src_key_padding_mask, base_tgt_key_padding_mask
    

    def get_tokens_batch(self):
        """
        Returns a matrix containing the tokens of all the instances in the batch
        and a list of the corresponding instance sizes.
        """

        token_batch = torch.zeros(self.num_instances,self.max_inst_size)

        for i,ins in enumerate(self.instances):
            tokens = self.pb.tokenize(ins)
            token_batch[i,:self.instance_lengths[i]] = tokens

        return token_batch, self.instance_lengths
    

    def get_ep_step(self):
        return self.curr_step % self.instance_lengths

    def get_finals(self):

        return self.curr_step % self.instance_lengths == 0





# TODO: check that config is valid
def load_config(path:Path):
    
    with open(path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)


    if not cfg['networks']['dim_embed'] % cfg['networks']['actor']['nhead']:
        raise ValueError('The number of heads needs to be a divisor of the embedding dimension')


    return cfg
