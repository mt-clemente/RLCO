from pathlib import Path
from einops import rearrange
from matplotlib import pyplot as plt
import torch
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import TensorDataset,DataLoader

from actor_critic import ActorCritic
from buffer import Buffer
from cop_class import COPInstance, COProblem
from environment import Environment

DEBUG = False



class PPOAgent(nn.Module):
    #TODO: 
    # - add scheduler support
    # - add separate process workers on top of simple parallel workers

    # FIXME: UNIT FIXME: UNIT FIXME: UNIT FIXME: UNIT FIXME: UNIT

    def __init__(self,
                 cfg:dict,
                 pb:COProblem,
                 instances:list[COPInstance],
                 max_num_segments,
                 buf:Buffer,
                 init_state = None,
                 eval_model_dir = None,
                 device = None,load_list=None
                 ):
        
        super().__init__()

        self.update_counter = 0
        self.train_cfg:dict = cfg['training']
        self.pb = pb
        self.instances = instances
        self.buf = buf
        self.value_weight = cfg['network']['value_weight']
        self.policy_weight = cfg['network']['policy_weight']
        self.entropy_weight = cfg['network']['entropy_weight']
        self.dim_embed = cfg['network']['dim_embed']
        self.cat_embedding_size = self.dim_embed
        net_cfg = cfg['network']

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        else:
            self.device = device

        if "grad_clip" in self.train_cfg.keys():
            self.grad_clip = self.train_cfg['grad_clip']
        
        else:
            self.grad_clip = False

        self.model = ActorCritic(
            cfg=net_cfg,
            num_segments=max_num_segments,
            max_inst_size=max([i.size for i in instances]),
            device=self.device
        )

        try:
            self.embedding = nn.Embedding(
                num_embeddings=self.pb.categorical_size,
                embedding_dim=self.cat_embedding_size-2,
                device=device
            )
            self.state_unit = torch.int

        except AttributeError:
            self.embedding = nn.Linear(
                in_features=self.pb.token_size,
                out_features=self.dim_embed,
                device=device
            )
            self.state_unit = torch.float

        if not eval_model_dir is None:
            self.load_model(eval_model_dir,load_list)

        if net_cfg['separate_value_training']:
            act_opt_cfg = net_cfg['actor']['optimizer']
            self.actor_optimizer = self.init_optimizer(self,opt_cfg=act_opt_cfg)
            crt_opt_cfg = net_cfg['actor']['optimizer']
            self.critic_optimizer = self.init_optimizer(self,opt_cfg=crt_opt_cfg)

        else:
             self.optimizer = self.init_optimizer(opt_cfg=net_cfg['optimizer'])

    def init_optimizer(self,opt_cfg) -> torch.optim.Optimizer:

        if 'type' in opt_cfg.keys():
            match 'type':
                case 'Adam':
                    optimizer = torch.optim.AdamW

                case 'SGD':
                    optimizer = torch.optim.SGD

                case 'RMSProp':
                    optimizer = torch.optim.RMSprop
        
        else:
            optimizer = torch.optim.AdamW

        combined_params = list(self.model.parameters()) + list(self.embedding.parameters())

        return optimizer(combined_params,**opt_cfg)


    def load_model(self,eval_model_dir:Path,load_list:list):
        raise NotImplementedError

    def tokenize(self, states:torch.Tensor,segments:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        
        state_embeddings = torch.cat((
            states[:,:2],
            self.embedding(states[:,2].int())
        ),
        -1)
        segment_embeddings = torch.cat((
            segments[:,:,:2],
            self.embedding(segments[:,:,2].int())
        ),
        -1)

        state_tokens,segment_tokens = self.pb.to_tokens(state_embeddings,segment_embeddings)

        return state_tokens.to(self.device), segment_tokens.to(self.device)

    def get_policy(
            self,
            state_tokens,
            valid_action_mask,
            ):


        policy = self.model.get_policy(
            state_tokens=state_tokens,
            valid_action_mask=valid_action_mask,
        )

        return policy
    
    def compute_gae_rtg(self,  buf:Buffer, gamma, gae_lambda,segments,src_key_padding_masks):

        # FIXME: Timesteps
        # State dims are [instance,step,state,token]
        states = torch.cat((buf.state_buf,buf.horzion_states.unsqueeze(1)),dim=1).transpose(0,1) # --> [step, instance, state, token]
        # states = buf.state_buf.transpose(0,1) # --> [step, instance, state, token]
        timesteps = torch.cat((buf.timestep_buf,buf.horzion_timesteps.unsqueeze(1).to(self.device)),dim=1).transpose(0,1).unsqueeze(-1)
        rewards = buf.rew_buf
        finals = buf.final_buf

        value_steps = []
        for i, (state_step, timestep_step) in enumerate(zip(states, timesteps)):
            embedded_state_step,embedded_segment_step = self.tokenize(state_step,segments)

            value_step = self.model.critic.forward(
                embedded_state_step,
            )

            value_steps.append(value_step.squeeze())

        values = torch.stack(value_steps, dim=1)
        next_values = values[:,1:].detach()
        values = values[:,:-1]


        # FIXME: Check if there is a problem with the new shapes
        td_errors = rewards + gamma * next_values * (1 - finals) - values.detach()
        gae = 0
        advantages = torch.zeros_like(td_errors)

        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + gamma * gae_lambda * (1 - finals[t]) * gae
            advantages[t] = gae

        returns_to_go = torch.zeros_like(rewards)
        return_to_go = next_values[:,-1] * (1-finals[:,-1])
        returns_to_go[:,-1] = return_to_go
        for t in reversed(range(rewards.size(-1)-1)):
            return_to_go = rewards[:,t] + gamma * (1 - finals[:,t]) * return_to_go
            returns_to_go[:,t] = return_to_go 

        return advantages, returns_to_go,values

    def rollout(self,env:Environment):

        (
            states,
            segments,
            action_masks,
            steps

        ) = env.get_training_state()

        t0 = datetime.now()

        for _ in range(env.horizon):

            with torch.no_grad():

                state_tokens, segment_tokens = self.tokenize(states,segments)
                policy = self.get_policy(
                    state_tokens=state_tokens.squeeze(),
                    valid_action_mask=action_masks,
                )
            actions = torch.multinomial(policy,1).squeeze()
            probs = policy[torch.arange(policy.size(0)),actions]
            

            new_states, _, rewards, done, new_action_masks, new_steps = env.step(actions)
            self.buf.push(
                state=states,
                policy=probs,
                action=actions,
                mask=action_masks,
                reward=rewards,
                ep_step=steps,
                final=done
            )

            
            if True in done:
                new_states, new_action_masks = env.reset(done) # Only reset where done is True

            
            action_masks = new_action_masks
            states = new_states
            steps = new_steps

                
        self.buf.push_horizon(
            state=states,
            step=steps,
        )


        print(env.curr_step)


        print(f"Rollout {self.update_counter} - log total gathered {torch.log10(torch.tensor(env.curr_step*states.size(0))):.2f} : {datetime.now()-t0}")
        self.update_counter += 1


    def update(self,env:Environment):
        """
        Updates the ppo agent, using the trajectories in the memory buffer.
        For states, policy, rewards, advantages, and timesteps the data is in a 
        straightforward format [batch,*values]
        For the returns-to-go and actions the data has a format [batch,sequence_len+1].
        We need the sequence coming before the state to make a prediction, and the current
        action to calculate the policy and ultimately the policy loss.

        """

        t0 = datetime.now()

        advantages, returns,values = self.compute_gae_rtg(
            buf=self.buf,
            gamma=self.train_cfg['gamma'],
            gae_lambda=self.train_cfg['gae_lambda'],
            segments=env.segments,
            src_key_padding_masks=env.src_key_padding_masks
        )


        dataset = TensorDataset(
            rearrange(self.buf.state_buf,'i s t -> (i s) t'),
            rearrange(self.buf.act_buf,'i s -> (i s)'),
            rearrange(values,'i s -> (i s)'),
            rearrange(self.buf.mask_buf,'i s m -> (i s) m'),
            rearrange(self.buf.policy_buf,'i s -> (i s)'),
            rearrange(advantages,'i s -> (i s)'),
            rearrange(returns,'i s -> (i s)'),
            torch.arange(env.num_instances,device=env.device).repeat_interleave(self.buf.act_buf.size(1))
        )

        loader = DataLoader(dataset, batch_size=self.train_cfg['minibatch_size'], shuffle=True, drop_last=False)

        # Perform multiple update epochs
        for k in range(self.train_cfg['epochs']):
            for batch in loader:


                (
                    batch_states,
                    batch_actions,
                    batch_values,
                    batch_action_masks,
                    batch_old_policies,
                    batch_advantages,
                    batch_returns,
                    instance_ids
                    
                ) = batch
                
                embedded_states, embedded_segments = self.tokenize(batch_states,env.segments[instance_ids])


                batch_values = self.model.critic(
                    embedded_states,
                )
                
                batch_policy = self.model.actor.forward(
                    embedded_states,
                    invalid_action_mask=batch_action_masks

                )

                # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
                # batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-4)

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1)).squeeze()
                ratio = action_probs / (batch_old_policies + 1e-5)
                clipped_ratio = torch.clamp(ratio, 1 - self.train_cfg['clip_eps'], 1 + self.train_cfg['clip_eps'])
                surrogate1 = ratio * batch_advantages.unsqueeze(1)
                surrogate2 = clipped_ratio * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surrogate1, surrogate2).mean() * self.policy_weight
                # Calculate value function loss
                value_loss = F.mse_loss(batch_values.squeeze(-1), batch_returns) * self.value_weight
                
                # Calculate entropy bonus
                entropy = -(batch_policy * torch.log(batch_policy+1e-6)).sum(-1).mean()

                entropy_loss = -self.entropy_weight * entropy
                # Compute total loss and update parameters

                if self.train_cfg['separate_value_training']:
                    pol_loss = policy_loss + entropy_loss
                    val_loss = value_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    pol_loss.backward()
                    val_loss.backward()

                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                else:
                    
                    loss = policy_loss + value_loss + entropy_loss
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)
                    
                    self.optimizer.step()
                  
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


        print("Update : ",datetime.now()-t0)

        with torch.no_grad():


            entropy = -(batch_policy * torch.log(batch_policy+1e-6)).sum(-1)

            max_entropy = torch.log(torch.logical_not(batch_action_masks).sum(-1))
            max_entropy = max_entropy[torch.logical_not(batch_action_masks).sum(-1) != 1]
            entropy = entropy[torch.logical_not(batch_action_masks).sum(-1) != 1]
            rel_entropy = entropy / max_entropy

        wandb.log({
            # "Current learning rate":self.scheduler.get_last_lr()[0],
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
            "Relative Entropy":rel_entropy.mean(),
            "Returns":returns.mean(0),
            "Reward repartition":self.buf.rew_buf.mean(0),
            "Mean buffer reward":self.buf.rew_buf.mean(),
            "Advantage repartition":advantages.mean(0),
            "Value repartition":batch_values.squeeze(-1).detach(),
            # "Total KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-5) - torch.log(batch_policy + 1e-5))).sum(dim=-1).mean()
            })


        self.buf.reset()