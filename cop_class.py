from abc import ABC, abstractmethod
from pathlib import Path
import torch

class COPInstance():

    def __init__(self,state,segments,**kwargs) -> None:
        self.state = state
        self.segments = segments
        self.kwargs = kwargs

class COPInstanceBatch():
    """
    TODO: Catch specific errors
    """

    def __init__(self,instances:list[COPInstance],device) -> None:

        self.states = [i.state for i in self.instances]
        self.segments = [i.segments for i in self.instances]
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


    def __len__(self):
        return len(self.states)

class COProblem(ABC):
    """
    Abstract class to define Combinatorial Optimization problems.
    If parallel is set to True, the rollouts will be able to be processed at the same time
    on one compute node. This enables us to have separate environements with optimal memory
    usage, as we do not have to store multiple workers, meaning that we only store the model
    parameters once.

    The 'parallel' flag is true by default. If the problem state can be represented as a
    tensor, and the states can be incremented in parallel it provides a huge speedup in 
    trajectory collection. If not, specify parllel=False, but this will slow down training
    significantly.

    """

    def __init__(self, dim_embedding, device=None,parallel=True) -> None:
        self.dim_embedding = dim_embedding
        self.parallel = parallel

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.num_segments = None
        self.instance = None



    def load_instance(self,path:Path):
        instance = self._load_instance(path)

        if self.num_segments is None:
            raise ValueError("The max number of solution segments has not been set")
        
        if isinstance(instance.state, torch.Tensor) and not self.parallel:
            print(UserWarning("The instance states are torch tensors, but the parallel"
            " flag is set to False. If the actions can be parallelized, change parallel to"
            " True to speed up training"))

        return instance


    @abstractmethod
    def _load_instance(self,path:Path) -> COPInstance:
        """
        Loads the instance from the specified path.
        Needs to return a list of solution segments
        """
        raise NotImplementedError
    



    

    @abstractmethod
    def act(self,instances:COPInstanceBatch,**kwargs) -> tuple():
        """
        Increments the current state of the solution by adding one segment chosen
        based on the current policy. It takes in the action and the state and returns a tuple:
        new_state, reward
        
        """

        #FIXME: IMPLEMENT token passing to act
        raise NotImplementedError


    

    @abstractmethod
    def tokenize(self,instance:COPInstanceBatch) -> torch.Tensor:
        """
        This method takes in the instance of the problem and outputs tokens
        that can be processed by a transformer.

        Returns the tokenized states and segments as sequences of tokens:
        If you have a batch of states representing a 2D grid of shape [batch, h, w] you
        want to return a tensor of shape [batch, h*w, dim_embed]

        TODO: If the size of the embedding does not correspond to the dim of a pretrained model you want to use,
        it is possible to use a conv1d with the correct number of channels trained like an autoencoder to 
        adapt the size of embedddings
        """
        
        if isinstance(instance.states,torch.Tensor) and isinstance(instance.segments,torch.Tensor) \
            and instance.states.size(-1) == self.dim_embedding and instance.segments.size(-1) == self.dim_embedding:

            return instance
    
        else:
            raise NotImplementedError



    @abstractmethod
    def valid_action_mask(self,state) -> torch.BoolTensor:
        """
        Outputs a mask that is True if an action is valid from the current state,
        and false if it is not.
        """





class InstanceBatchManager():

    def __init__(self,instances_path:Path,problem:COProblem) -> None:

        self.pb = problem

        
        instances = []
        for file in instances_path.iterdir():
            inst = self.pb.load_instance(file)
            instances.append(inst)

            if max_inst_size < len(inst):
                max_inst_size = len(inst)

        self.instances = instances

        if problem.parallel:
            self.states = torch.vstack([i.state for i in self.instances])
            self.segments = torch.vstack([i.segments for i in self.instances])


        self.num_instances = len(instances)
        self.max_inst_size = max_inst_size
        self.instance_lengths = torch.tensor([len(x) for x in self.instances]).unsqueeze(0)



    def get_padding_masks(self):

        # pad to get all max length sequences
        src_key_padding_mask = torch.arange(self.max_inst_size) > self.instance_lengths
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



