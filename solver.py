from pathlib import Path
import torch
import torch
from buffer import Buffer
from cop_class import COProblem
from ppo import PPOAgent
from utils import TrainingManager, load_config





class RLCOSolver():

    # The passed problem is a

    # TODO: 
    #  - look at where to put the init state
    #  - add separate worker processes
    #  - add different types of policy sampling (prob, max, topk, etc.)


    def __init__(self, config_path:Path,instances_path:Path,problem:COProblem,eval_model_dir:Path=None,load_list:list=None) -> None:

        self.cfg = load_config(config_path)

        try:
            pb:COProblem = problem(self.cfg['problem'])

        except KeyError:
            pb:COProblem = problem(self.cfg['network']['dim_embed'])

        device = 'cuda' if torch.cuda.is_available() and (self.cfg['cuda']) else 'cpu'

        self.manager = TrainingManager(self.cfg['training'],instances_path,pb,device=device)

        (
            num_instances,
            max_inst_size,
            max_num_segments,
            dim_token

        ) = self.manager.batch_attributes()

        buf = Buffer(
            cfg=self.cfg,
            num_instances=num_instances,
            max_ep_len=max_inst_size,
            max_num_segments=max_num_segments,
            dim_token=dim_token,
            device=device,
        )

        self.agent = PPOAgent(
            cfg=self.cfg,
            buf=buf,
            max_instance_size=self.manager.max_inst_size,
            eval_model_dir=eval_model_dir,
            pb=pb,
            device=device,
            load_list=load_list
        )



    def train(self):

        try:
            while self.manager.stop_criterion():

                self.agent.rollout(self.manager)
                self.agent.update(self.manager)
                
                if self.manager.episode % self.cfg['checkpoint_period']:
                    #TODO: Save models
                    pass
        
        except KeyboardInterrupt:
            pass

        # TODO: save solution