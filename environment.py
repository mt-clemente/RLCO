import copy
from pathlib import Path
import torch
import yaml
from cop_class import COProblem
import torch.nn.functional as F
import warnings
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

    def __init__(self,cfg,instance_path:Path,problem:COProblem,device) -> None:

        self.device = device
        self.pb = problem
        self.cfg = cfg

        # Batch mgt
        instances = []

        if instance_path.is_dir():

            if cfg['num_workers'] < len(list(instance_path.iterdir())):
                warnings.warn('Number of workers lower than number of instances in the given directory, only loading the first n_w files',UserWarning,stacklevel=2)
            
            for i,file in enumerate(instance_path.iterdir()):
                inst = self.pb.load_instance(file)
                instances.append(inst)

                if i >= cfg['num_workers']:
                    break

        else:

            if cfg['num_workers'] is None:
                warnings.warn("Number of workers not in config, defaulting to 1",UserWarning)
                cfg['num_workers'] = 1

            for _ in range(cfg['num_workers']):
                inst = self.pb.load_instance(instance_path)
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
        self.base_masks = torch.logical_not(self.src_key_padding_masks)
        self.masks = self.base_masks.clone()
        self.causal_mask = torch.triu(torch.ones(self.max_inst_size, self.max_inst_size,device=self.device), diagonal=1)

        self.best_sol = None
        self.best_sol_eval = None
        self.best_perf = None

        self.cfg = self.cfg
        self.episode = torch.zeros_like(self.sizes)
        self.curr_step = 0
        self.horizon = self.cfg['horizon']


        
    def stop_criterion(self):

        stop_cfg = self.cfg['stop_criterion']

        match stop_cfg['type']:

            case 'inf':
                return True
            
            case 'ep':
                return self.episode > stop_cfg['n_ep']
            


    def get_training_state(self):

        steps = self.get_ep_step()
        action_masks = self.pb.valid_action_mask_(self.states,self.segments,self.masks) 

        return self.states, self.segments, action_masks, steps
    

    def finalize_rollout(self,states):
        self.states = states

    def init_masks(self):
        
        # pad to get all max length sequences
        src_key_padding_mask = torch.arange(self.max_num_segments).expand((self.num_instances,-1)) >= self.instance_num_segments.unsqueeze(-1)
        
        return src_key_padding_mask.to(self.device)
    
    def get_tokens(self):
        """
        Returns a matrix containing the tokens of all the instances in the batch
        and a list of the corresponding instance sizes.
        """

        state_tokens, segment_tokens = self.pb.tokenize(self.states,self.segments)

        return state_tokens.to(self.device), segment_tokens.to(self.device)
    

    def get_ep_step(self):
        return self.curr_step % self.ep_len

    def get_dones(self):

        return (self.curr_step) % (self.ep_len) == 0

    def batch_attributes(self):
        return (
            self.num_instances,
            self.max_inst_size,
            self.max_num_segments,
            self.dim_token
        )



    def step(self,actions:torch.Tensor):

        added_tokens = self.segments[torch.arange(self.segments.size(0),device=self.segments.device),actions]
        if 0 in added_tokens.sum(-1).squeeze():
            raise ValueError("Null token chosen: in the following segments : ",added_tokens,actions,self.sizes)
        
        new_states, rewards = self.pb.act(self.states,added_tokens,self.get_ep_step(),self.sizes)

        self.curr_step += 1

        dones = self.get_dones()
        steps = self.get_ep_step()
        self.masks[torch.arange(actions.size(0)),actions] = False
        self.states = new_states

        # if self.pb.verify_solution(self.states[0],self.segments[0],self.sizes[0]):
            # print(self.curr_step)
        
        # self.states[dones] = self.init_states[dones]
        # self.masks[dones] = self.base_masks[dones]
        action_masks = self.pb.valid_action_mask_(self.states,self.segments,self.masks) 

        return self.states, self.segments, rewards, dones, action_masks, steps



    def reset(self,dones=None):

        if dones is None:
            self.states = self.init_states
            self.masks = self.base_masks
        
        else:
            self.states[dones] = self.init_states[dones]
            self.masks[dones] = self.base_masks[dones]

        return self.states,self.pb.valid_action_mask_(self.states,self.segments,self.masks) 





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

    if not 'num_workers' in cfg['training'].keys():
        cfg['training']['num_workers'] = None

    return cfg

