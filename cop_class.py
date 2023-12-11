from abc import ABC, abstractmethod
from pathlib import Path
import torch

class COPInstance():

    def __init__(self,state,segments,size,num_segments,**kwargs) -> None:
        self.state = state
        self.segments = segments
        self.kwargs = kwargs
        self.size = size
        self.num_segments = num_segments

class COPInstanceBatch():
    """
    TODO: Catch specific errors
    """

    def __init__(self,instances:list[COPInstance],device) -> None:

        self.states = [i.state for i in instances]
        self.segments = [i.segments for i in instances]
        self.device = device

        # All the istances are supposed to be of the same problem,
        # hence all the kwargs have the same keys.

        self.kwargs = {}

        for key in instances[0].kwargs.keys():

            try:
                self.kwargs[key] = torch.tensor([i.kwargs[key] for i in instances],device=device)

            except:
                self.kwargs[key] = [i.kwargs[key] for i in instances]


        try:
            self.states = torch.vstack(self.states)
            self.segments = torch.vstack(self.segments)

        except:
            pass

class COProblem(ABC):
    """
    Abstract class to define Combinatorial Optimization problems.

    """

    def __init__(self, dim_embedding, cfg=None, device=None,reuse_segments=False) -> None:

        self.cfg = cfg
        self.dim_embedding = dim_embedding
        self.reuse_segments = reuse_segments

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_segments = None
        self.instance = None



    def load_instance_(self,path:Path):
        instance = self.load_instance(path)

        if self.num_segments is None:
            raise ValueError("The max number of solution segments has not been set")
        
        return instance


    @abstractmethod
    def load_instance(self,path:Path) -> COPInstance:
        """
        Loads the instance from the specified path.
        Needs to return a list of solution segments
        """
        raise NotImplementedError
    

    @abstractmethod
    def act(self,states,segments,steps,**kwargs) -> tuple:
        """
        Increments the current state of the solution by adding one segment chosen
        based on the current policy. It takes in the action and the state and returns a tuple:
        new_state, reward
        
        """

        raise NotImplementedError



    def valid_action_mask(self,states:torch.Tensor,segments:torch.Tensor) -> torch.BoolTensor:
        """
        Outputs a mask that is True if an action is valid from the current state,
        and false if it is not.
        """
        
        return torch.ones_like(segments[0,:,0]) == 1

    def valid_action_mask_(self,states:torch.Tensor,segments:torch.Tensor,used_mask:torch.Tensor=None):

        if self.reuse_segments:
            return self.valid_action_mask(states,segments)

        else:
            return torch.logical_not(torch.logical_and(self.valid_action_mask(states,segments),used_mask))


    @abstractmethod
    def to_tokens(self,states,segments):
        raise NotImplementedError

    def display_solution(self,solution,file):
        raise NotImplementedError


# class InstanceBatchManager():

#     def __init__(self,instances_path:Path,problem:COProblem) -> None:

#         self.pb = problem
        
#         instances = []
#         for file in instances_path.iterdir():
#             inst = self.pb.load_instance(file)
#             instances.append(inst)

#         self.instances = instances

#         if isinstance(self.states[0],torch.Tensor) and isinstance(self.segments[0],torch.Tensor):
#             self.states = torch.vstack([i.state for i in self.instances])
#             self.segments = torch.vstack([i.segments for i in self.instances])
#             self.sizes= torch.tensor([i.size for i in self.instances])


#         self.num_instances = len(instances)
#         self.max_inst_size = max([i.size for i in self.instances])
#         self.instance_lengths = torch.tensor([len(x) for x in self.instances]).unsqueeze(0)



#     def get_padding_masks(self):

#         # pad to get all max length sequences
#         src_key_padding_mask = torch.arange(self.max_inst_size) > self.instance_lengths
#         base_tgt_key_padding_mask = src_key_padding_mask.clone()


#         return src_key_padding_mask, base_tgt_key_padding_mask
    

#     def get_tokens_batch(self):
#         """
#         Returns a matrix containing the tokens of all the instances in the batch
#         and a list of the corresponding instance sizes.
#         """

#         token_batch = torch.zeros(self.num_instances,self.max_inst_size)

#         for i,ins in enumerate(self.instances):
#             tokens = self.pb.tokenize(ins)
#             token_batch[i,:self.instance_lengths[i]] = tokens

#         return token_batch, self.instance_lengths



