import copy
from pathlib import Path
from einops import rearrange, repeat
import torch
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import TensorDataset,DataLoader

from actor_critic import ActorCritic
from buffer import Buffer
from cop_class import COProblem
from utils import TrainingManager, load_config

DEBUG = False








class PPOAgent(nn.Module):
    #TODO: 
    # - add scheduler support
    # - add separate process workers on top of simple parallel workers

    # FIXME: UNIT FIXME: UNIT FIXME: UNIT FIXME: UNIT FIXME: UNIT

    def __init__(self,
                 cfg:dict,
                 max_instance_size,pb:COProblem,
                 buf:Buffer,
                 init_state = None,
                 eval_model_dir = None,
                 device = None,load_list=None
                 ):
        
        super().__init__()

        self.train_cfg:dict = cfg['training']
        self.pb = pb
        self.buf = buf

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        else:
            self.device = device

        if "grad_clip" in self.train_cfg.keys():
            self.grad_clip = self.train_cfg['grad_clip']
        
        else:
            self.grad_clip = False

        self.model = ActorCritic(
            cfg=cfg['network'],
            num_segments=max_instance_size,
            init_state=init_state,
            device=self.device
        )

        if not eval_model_dir is None:
            self.load_model(eval_model_dir,load_list)

    def load_model(self,eval_model_dir:Path,load_list:list):
        raise NotImplementedError


    def get_policy(self, states, segments, timesteps, valid_action_mask,src_key_padding_masks):

        state_tokens, segment_tokens = self.pb.tokenize(states,segments)

        policy = self.model.get_policy(
            state_tokens=state_tokens,
            segment_tokens=segment_tokens,
            timesteps=timesteps,
            valid_action_mask=valid_action_mask,
            src_key_padding_masks=src_key_padding_masks
        )

        return policy
    
    def compute_gae_rtg(self,  buf:Buffer, gamma, gae_lambda,segments,src_key_padding_masks):

        # if (self.cur) % self.horizon != 0:
        #     raise BufferError("Calculating GAE at wrong time")
        #     pass

        # State dims are [instance,step,state,token]
        states = torch.cat((buf.state_buf,buf.horzion_states),dim=1)
        timesteps = torch.cat((buf.timestep_buf,buf.horzion_timesteps))
        rewards = buf.rew_buf
        finals = buf.final_buf

        values = torch.tensor((states.size(0),states.size(1)+1))

        for i, (state_step, timestep_step) in enumerate(zip(states, timesteps)):

            src_inputs, tgt_inputs,  tgt_key_padding_mask = self.model.make_transformer_inputs(
                states=state_step,
                segments=segments,
                timesteps=timestep_step
            )

            value_step = self.model.critic.forward(
                src_inputs=src_inputs,
                tgt_inputs=tgt_inputs,
                timesteps=timestep_step,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_masks
            )

            values[i] = value_step


        next_values = values[:,1:]
        values = values[:,:-1]


        # FIXME: Check if there is a problem with the new shapes


        td_errors = rewards + gamma * next_values * (1 - finals) - values
        gae = 0
        advantages = torch.zeros_like(td_errors)

        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + gamma * gae_lambda * (1 - finals[t]) * gae
            advantages[t] = gae


        returns_to_go = torch.zeros_like(rewards)
        return_to_go = next_values[-1]
        returns_to_go[-1] = return_to_go
        for t in reversed(range(len(rewards)-1)):
            return_to_go = rewards[t] + gamma * (1 - finals[t]) * return_to_go
            returns_to_go[t] = return_to_go 

        advantages = advantages
        returns_to_go = returns_to_go

        return advantages, returns_to_go

    def rollout(self,manager:TrainingManager):

        (
            states,
            segments,
            steps

        ) = manager.get_training_state()


        for _ in range(manager.horizon):
            
            
            action_masks = self.pb.valid_action_mask_(states,segments,manager.masks)

            policy = self.get_policy(
                states=states,
                segments=segments,
                timesteps=steps,
                valid_action_mask=action_masks,
                src_key_padding_masks=manager.src_key_padding_masks,
            )

            new_states, rewards, probs, actions = self.increment_states(
                states=states,
                segments=segments,
                policies=policy,
                steps=steps
            )


            self.buf.push(
                state=states,
                policy=probs,
                action=actions,
                mask=action_masks,
                reward=rewards,
                ep_step=steps,
                final=manager.get_finals()
            )
        
            states = new_states

            # update finals / masks based on actions taken
            manager.step(actions)

        # inject the states at the end of the horizon, needed to calculate advantages
        manager.finalize_rollout(states)
        self.buf.horzion_states = states
        self.buf.horzion_timesteps = steps + 1 % manager.instance_lengths

        
    def increment_states(self,states,segments:torch.Tensor,policies:torch.Tensor,steps):
        """
        Adds the tokens selected by the policies to the current states.
        Returns new states, associated rewards, probabilities for each action w.r.t the current
        policy, and the actions (chosen action indexes).

        TODO: For the parallel version, the states are tensors, might need to add support for Any 
        types
        """

        actions = torch.multinomial(policies,1)
        added_tokens = segments[torch.arange(segments.size(0),device=segments.device),actions]
        new_states, rewards = self.pb.act(states,added_tokens,steps)

        return new_states, rewards, policies[actions], actions


    def update(self,manager:TrainingManager):
        """
        Updates the ppo agent, using the trajectories in the memory buffer.
        For states, policy, rewards, advantages, and timesteps the data is in a 
        straightforward format [batch,*values]
        For the returns-to-go and actions the data has a format [batch,sequence_len+1].
        We need the sequence coming before the state to make a prediction, and the current
        action to calculate the policy and ultimately the policy loss.

        """

        t0 = datetime.now()

        advantages, returns = self.compute_gae_rtg(
            buf=self.buf,
            gamma=self.train_cfg['gamma'],
            gae_lambda=self.train_cfg['gae_lambda'],
            segments=manager.segments,
            src_key_padding_masks=manager.src_key_padding_masks
        )

        dataset = TensorDataset(
            rearrange(self.buf.state_buf,'i s p t -> (i s) p t'),
            rearrange(self.buf.act_buf,'i s -> (i s)'),
            rearrange(self.buf.mask_buf,'i s m -> (i s) m'),
            rearrange(self.buf.policy_buf,'i s m -> (i s) m'),
            rearrange(self.buf.timestep_buf,'i s -> (i s)'),
            rearrange(advantages,'i s -> (i s)'),
            rearrange(returns,'i s -> (i s)'),
            torch.arange(manager.num_instances,device=manager.device).repeat_interleave(self.buf.state_buf.size(1))
        )


        loader = DataLoader(dataset, batch_size=self.train_cfg['minibatch_size'], shuffle=True, drop_last=False)

        # Perform multiple update epochs
        for k in range(self.train_cfg['epochs']):
            for batch in loader:

                (
                    batch_states,
                    batch_actions,
                    batch_masks,
                    batch_old_policies,
                    batch_timesteps,
                    batch_advantages,
                    batch_returns,
                    instance_ids
                    
                ) = batch

                # FIXME: self. wtf??

                src_inputs, tgt_inputs,  tgt_key_padding_mask = self.model.make_transformer_inputs(
                    batch_states,
                    manager.segments[instance_ids],
                    batch_timesteps,
                )
                
                batch_policy, batch_value = self.model.actor.forward(
                    src_inputs=src_inputs,
                    tgt_inputs=tgt_inputs,
                    timesteps=batch_timesteps,
                    src_key_padding_mask=manager.src_key_padding_masks[instance_ids],
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    valid_action_mask=batch_masks

                )

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-4)

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1))
                old_action_probs = batch_old_policies.gather(1, batch_actions.unsqueeze(1))
                ratio = action_probs / (old_action_probs + 1e-5)
                clipped_ratio = torch.clamp(ratio, 1 - self.train_cfg['clip_eps'], 1 + self.train_cfg['clip_eps'])
                surrogate1 = ratio * batch_advantages.unsqueeze(1)
                surrogate2 = clipped_ratio * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                # Calculate value function loss
                value_loss = F.mse_loss(batch_value.squeeze(-1), batch_returns) * self.value_weight

                # Calculate entropy bonus
                entropy = -(batch_policy[batch_policy != 0] * torch.log(batch_policy[batch_policy != 0])).sum(dim=-1).mean()
                entropy_loss = -self.train_cfg['entropy_weight'] * entropy
                # Compute total loss and update parameters



                
                if self.train_cfg['separate_value_training']:

                    pol_loss = policy_loss + entropy_loss
                    val_loss = value_loss

                    self.model.actor_optimizer.zero_grad()
                    self.model.critic_optimizer.zero_grad()

                    pol_loss.backward()
                    val_loss.backward()

                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)

                    self.model.actor_optimizer.step()
                    self.model.critic_optimizer.step()

                else:
                    
                    loss = policy_loss + value_loss + entropy_loss
                    self.model.optimizer.zero_grad()
                    loss.backward()
                  
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)
                  
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
                    print("Loss",entropy_loss.item(),value_loss.item(),policy_loss.item())
                    print("ratio",ratio.max())
                    print("ratio",ratio.min())
                    # print("ratio",((ratio > 1 + self.ac_cfg['clip_eps']).count_nonzero() + (ratio < 1 - self.ac_cfg['CLIP_EPS']).count_nonzero()))
                    print("bst",batch_states.max())
                    print("bst",batch_states.min())
                    self.model.optimizer.step()


        print(datetime.now()-t0)
        wandb.log({
            "Current learning rate":self.scheduler.get_last_lr()[0],
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
            "Value repartition":batch_value.squeeze(-1).detach(),
            "Total KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-5) - torch.log(batch_policy + 1e-5))).sum(dim=-1).mean()
            })


        # for worker in self.workers:
        #     worker.load_state_dict(self.model.state_dict())




