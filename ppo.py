import copy
from pathlib import Path
import torch
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from actor_critic import ActorCritic
from buffer import Buffer
from cop_class import InstanceBatchManager, COProblem
from utils import TrainingManager, load_config

DEBUG = False

class PPOTrainer():

    # TODO: 
    #  - look at where to put the init state
    #  - add separate worker processes
    #  - add different types of policy sampling (prob, max, topk, etc.)


    def __init__(self, config_path:Path,problem:COProblem,eval_model_dir:Path=None,load_list:list=None) -> None:

        self.cfg = load_config(config_path)
        self.pb = problem
        device = 'cuda' if torch.cuda.is_available() and (self.cfg['cuda']) else 'cpu'

        max_inst_size = 0

        self.agent = PPOAgent(
            cfg=self.cfg,
            max_instance_size=max_inst_size,
            eval_model_dir=eval_model_dir,
            device=device,
            load_list=load_list
        )

        self.buf = Buffer(
            cfg=self.cfg,
            ep_len=max_inst_size,
            device=device,
        )


    def train(self,instances_path:Path):


        instance_batch = InstanceBatchManager(instances_path,self.pb)
        manager = TrainingManager(instance_batch)

        try:
            while manager.stop_criterion():

                current_state = self.rollout(manager)
                self.agent.update()
                
                manager.step(current_state)

                if manager.episode % self.cfg['checkpoint_period']:
                    #TODO: Save models
                    pass
        
        except KeyboardInterrupt:
            pass

        # TODO: save solution



    def rollout(self,manager:TrainingManager):

        (
            states,
            segments,
            step

        ) = manager.get_training_state()

        for _ in range(manager.horizon):
            
            action_mask = self.pb.valid_action_mask(states)

            policy = self.agent.get_policy(
                state_tokens=states,
                segment_tokens=segments,
                timesteps=step,
                valid_action_mask=action_mask, # TODO: just action mask
            )

            new_states, rewards, probs, actions = self.increment_states(
                states=states,
                tokens=segments,
                policy=policy
            )

            manager.step(new_states)

            self.buf.push(
                state=new_states,
                policy=probs,
                action=actions,
                reward=rewards,
                ep_step=manager.step,
                final=manager.get_final()
            )
        
        
    def increment_states(self,states,segments:torch.Tensor,policies:torch.Tensor):
        """
        Adds the tokens selected by the policies to the current states.
        Returns new states, associated rewards, probabilities for each action w.r.t the current
        policy, and the actions (chosen action indexes).

        TODO: For the parallel version, the states are tensors, might need to add support for Any 
        types
        """

        actions = torch.multinomial(policies,1)
        added_tokens = segments[torch.arange(segments,device=segments.device),actions]
        new_states, rewards = self.pb.act(states,added_tokens)

        return new_states, rewards, policies[actions], actions










class PPOAgent(nn.Module):
    #TODO: 
    # - add scheduler support
    # - add separate process workers on top of simple parallel workers

    def __init__(self,cfg, max_instance_size, init_state = None, eval_model_dir = None, device = None,load_list=None):
        super().__init__()

        self.train_cfg = cfg['training']

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        else:
            self.device = device

        self.model = ActorCritic(
            cfg['network'],
            init_state,
            num_segments=max_instance_size,
            device=device
        )

        if not eval_model_dir is None:
            self.load_model(eval_model_dir,load_list)


    def get_policy(self, state_tokens, segment_tokens, timesteps, valid_action_mask):

        policy = self.model.get_policy(
            state_tokens=state_tokens,
            segment_tokens=segment_tokens,
            timesteps=timesteps,
            valid_action_mask=valid_action_mask
        )

        return policy

    def update(self,loader:DataLoader):
        """
        Updates the ppo agent, using the trajectories in the memory buffer.
        For states, policy, rewards, advantages, and timesteps the data is in a 
        straightforward format [batch,*values]
        For the returns-to-go and actions the data has a format [batch,sequence_len+1].
        We need the sequence coming before the state to make a prediction, and the current
        action to calculate the policy and ultimately the policy loss.

        """

        t0 = datetime.now()

        # FIXME: calculate advantages / RTGs


        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:

                (
                    batch_states,
                    batch_actions,
                    batch_masks,
                    batch_advantages,
                    batch_old_policies,
                    batch_returns,
                    batch_timesteps,
                    
                ) = batch

                
                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_timesteps,
                    batch_masks,
                )

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-4)

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1))
                old_action_probs = batch_old_policies.gather(1, batch_actions.unsqueeze(1))
                ratio = action_probs / (old_action_probs + 1e-5)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                surrogate1 = ratio * batch_advantages.unsqueeze(1)
                surrogate2 = clipped_ratio * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                # Calculate value function loss
                value_loss = F.mse_loss(batch_value.squeeze(-1), batch_returns) * self.value_weight

                # Calculate entropy bonus
                entropy = -(batch_policy[batch_policy != 0] * torch.log(batch_policy[batch_policy != 0])).sum(dim=-1).mean()
                entropy_loss = -self.entropy_weight * entropy
                # Compute total loss and update parameters


                loss = policy_loss + value_loss + entropy_loss
                if DEBUG:
                    print(f"---- EPOCH {k} ----")
                    print("ba",batch_actions.max())
                    print("ba",batch_actions.min())
                    print("badv",batch_advantages.max())
                    print("badv",batch_advantages.min())
                    print("bop",batch_old_policies.max())
                    print("bop",batch_old_policies.min())
                    print("bret",batch_returns.max())
                    print("bret",batch_returns.min())
                    print("bp",batch_policy.max())
                    print("bp",batch_policy.min())
                    print("Loss",entropy_loss.item(),value_loss.item(),policy_loss.item(),loss)
                    print("ratio",ratio.max())
                    print("ratio",ratio.min())
                    print("ratio",((ratio > 1 + self.ac_cfg['clip_eps']).count_nonzero() + (ratio < 1 - self.ac_cfg['CLIP_EPS']).count_nonzero()))
                    print("bst",batch_states.max())
                    print("bst",batch_states.min())

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.optimizer.step()
                self.scheduler.step()
                g=0
                i = 0
                for name,param in self.model.named_parameters():
                    i+=1
                    # print(f"{name:>28} - {torch.norm(param.grad)}")
                    # print(f"{name:>28}")
                    g+=torch.norm(param.grad)

        print(datetime.now()-t0)
        wandb.log({
            "Total loss":loss,
            "Current learning rate":self.scheduler.get_last_lr()[0],
            "Cumul grad norm":g,
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
            "Value repartition":batch_value.squeeze(-1).detach(),
            "KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-5) - torch.log(batch_policy + 1e-5))).sum(dim=-1).mean()
            })


        # for worker in self.workers:
        #     worker.load_state_dict(self.model.state_dict())




