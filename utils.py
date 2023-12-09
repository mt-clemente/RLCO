import copy
from pathlib import Path
import torch
import yaml
from cop_class import COProblem
import torch.nn.functional as F

class Environment():
    """
    Used to monitor and manage the training.
    The current implementation does not cycle smaller implementations
    faster (we could do 2 episodes of a 2x smaller instance in the same
    nb of steps than the biggest instance of the batch), #TODO: for now
    it 'fills' the now empty threads with unfinished bigger instances.

    TODO: 
     - add performance stopping
     - add sequential action masks where the masks are calculated based
       on the previous mask, and not in 'closed form'
     - initial state mgt ? --> Directly in the COP Init
    """

    def __init__(self,cfg,instances_path:Path,problem:COProblem,device) -> None:

        self.device = device
        self.pb = problem
        self.cfg = cfg

        # Batch mgt
        instances = []
        for file in instances_path.iterdir():
            inst = self.pb.load_instance(file)
            instances.append(inst)

        self.instances = instances
        self.max_inst_size = max([i.size for i in self.instances])
        self.max_num_segments = max([len(i.segments) for i in self.instances])
        self.num_instances = len(instances)
        self.segment_size = self.instances[0].state.size(-1)

        # TODO: Init state mgt
        if isinstance(self.instances[0].state,torch.Tensor) and isinstance(self.instances[0].segments,torch.Tensor):

            self.states = torch.zeros((self.num_instances,self.max_inst_size,self.segment_size),device=device)
            self.segments = torch.zeros((self.num_instances,self.max_num_segments,self.segment_size),device=device)

            for i, instance in enumerate(self.instances):
                
                state = instance.state
                segments = instance.segments

                self.states[i,:instance.size] = state
                self.segments[i,:len(instance.segments)] = segments

            self.dim_token = self.states.size(-1)

                
        self.sizes= torch.tensor([i.size for i in self.instances])
        self.ep_len = torch.tensor([i.num_segments for i in self.instances])
        self.init_states = copy.deepcopy(self.states)

        self.instance_lengths = torch.tensor([x.size for x in self.instances])
        self.instance_num_segments = torch.tensor([x.num_segments for x in self.instances])

        self.src_key_padding_masks = self.init_masks()
        self.masks = self.src_key_padding_masks.clone()
        self.causal_mask = torch.triu(torch.ones(self.max_inst_size, self.max_inst_size,device=self.device), diagonal=1)

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

        stop_cfg = self.cfg['stop_criterion']

        match stop_cfg['type']:

            case 'inf':
                return True
            
            case 'ep':
                return self.episode > stop_cfg['n_ep']
            


    def get_training_state(self):

        return (
            self.states,
            self.segments
        )
    


    def step(self,actions):
        # self.states[finals] = self.init_states[finals]
        
        finals = self.get_finals()
        self.curr_step += 1
        self.masks[torch.arange(self.masks.size(0)),actions] = False

        # New episode
        self.masks[finals] = self.src_key_padding_masks[finals]


        # # FIXME:FIXME:FIXME:FIXME:FIXME:FIXME:FIXME:
        # if self.curr_step >= self.max_inst_size: #FIXME: +- 1?
        #     print(f"EPISODE {self.episode} : {self.best_perf}" )#FIXME: +- Custom perf?
        #     self.episode += 1
        #     self.reset()

    def finalize_rollout(self,states):
        self.states = states

    def init_masks(self):
        
        # pad to get all max length sequences
        src_key_padding_mask = torch.arange(self.max_num_segments).expand((self.num_instances,-1)) < self.instance_num_segments.unsqueeze(-1)
        
        return src_key_padding_mask.to(self.device)
    
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
        return self.curr_step % self.ep_len

    def get_finals(self):

        if self.curr_step <= 1:
            return torch.zeros_like(self.ep_len) == 1 # all False
        
        return (self.curr_step+1) % self.ep_len == 0

    def batch_attributes(self):
        return (
            self.num_instances,
            self.max_inst_size,
            self.max_num_segments,
            self.dim_token
        )




# TODO: check that config is valid
def load_config(path:Path):
    
    with open(path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    if cfg['network']['dim_embed'] % cfg['network']['actor']['nhead']:
        raise ValueError('The number of heads needs to be a divisor of the embedding dimension')
    
    match cfg['network']['unit']:

        case 'half':
            cfg['network']['unit'] = torch.half

        case 'double':
            cfg['network']['unit'] = torch.double

        case _ :
            cfg['network']['unit'] = torch.float

    if 'separate_value_training' not in cfg['network'].keys():
        cfg['network']['separate_value_training'] = False
        cfg['training']['separate_value_training'] = False
    else:
        cfg['training']['separate_value_training'] = cfg['network']['separate_value_training']

    return cfg
