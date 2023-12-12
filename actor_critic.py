import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MaskedStableSoftmax, Pointer, PositionalEncoding, PositionalEncoding1D, PositionalEncoding2D, Transformer

class ActorCritic(nn.Module):

    """
    Actor critic model.

    TODO:
    -----------------
    FIXME: UNIT MGT
    Add custom weight init
    """

    def __init__(self, cfg, num_segments,max_inst_size,device='cpu'):
        super().__init__()

        self.num_segments = num_segments
        self.device = device
        act_cfg = cfg['actor']

        self.actor = Actor(
            cfg=act_cfg,
            dim_embed=cfg['dim_embed'],
            num_segments=num_segments,
            device=device,
            unit=cfg['unit']
        )


        crt_cfg = cfg['critic']

        self.critic = Critic(
            cfg=crt_cfg,
            dim_embed=cfg['dim_embed'],
            device=device,
            unit=cfg['unit']
        )


        self.dim_embed = cfg['dim_embed'] 

        if 'categorical_vocab_size' in cfg.keys():
            self.embed_state = nn.Embedding(cfg['categorical_vocab_size'],self.dim_embed)
            self.embed_segment = nn.Embedding(cfg['categorical_vocab_size'],self.dim_embed)

        if cfg['pos_encoding'] == 1:
            self.positional_encoding = PositionalEncoding2D(cfg['dim_embed']) # FIXME: choose type
        elif cfg['pos_encoding'] == 2:
            self.positional_encoding = PositionalEncoding1D(cfg['dim_embed']) # FIXME: choose type
        else:
            self.positional_encoding = lambda x:0

            
        self.embed_state_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])
        self.embed_segment_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])


        self.causal_mask = torch.triu(torch.ones(max_inst_size, max_inst_size,device=self.device,dtype=bool), diagonal=1)



    def make_transformer_inputs(self,embedded_states,embedded_segments, timesteps):

        
        batch_size = embedded_states.shape[0]

        # FIXME: add positional encoding

        src_inputs = self.embed_segment_ln(embedded_segments)
        embedded_states += self.positional_encoding(embedded_states)
        tgt_inputs = self.embed_state_ln(embedded_states)
        tgt_key_padding_mask = torch.arange(self.num_segments+1,device=timesteps.device).repeat(batch_size,1) > timesteps
        return src_inputs, tgt_inputs, tgt_key_padding_mask.to(self.device)



    def get_policy(self, state_tokens,  valid_action_mask):


        policy_pred = self.actor.forward(
                src_inputs=state_tokens,
                invalid_action_mask=valid_action_mask,
        )

        return policy_pred

    
    def get_action(self, policy:torch.Tensor):
        return torch.multinomial(policy,1)
    




class Actor(nn.Module):


    def __init__(self, cfg, dim_embed, num_segments, device, unit) -> None:
        super().__init__()

        self.num_segments = num_segments        
        self.ptrnet = cfg['pointer']
        self.dim_embed=dim_embed

        self.mlp =  nn.Sequential(
            nn.Linear(dim_embed,cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],dim_embed,device=device),
        )
    

        if self.ptrnet:
            self.policy_attn_head = Pointer(
                dim_embed,
                device,
                unit,
                )

        else:
            self.actor_head = nn.Sequential(
                nn.GELU(),
                nn.Linear(dim_embed,num_segments,device=device,dtype=unit),
            )
            self.policy_head = MaskedStableSoftmax()


    
    def forward(self,src_inputs,invalid_action_mask):
        
        #FIXME: mask
        policy_token = self.mlp(
            src_inputs
        )

        policy_logits = self.actor_head(policy_token)
        policy_pred = self.policy_head(policy_logits,invalid_action_mask)

        return policy_pred


class Critic(nn.Module):

    def __init__(self, cfg, dim_embed, device, unit) -> None:
        super().__init__()

        self.mlp =  nn.Sequential(
            nn.Linear(dim_embed,cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],cfg['hidden_size'],device=device),
            nn.ReLU(),
            nn.Linear(cfg['hidden_size'],dim_embed,device=device),
        )
    
    
    
        self.critic_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_embed, 1,device=device,dtype=unit),
        )

        self.dim_embed = dim_embed



    def forward(self,src_inputs):
        
        value_token = self.mlp(
            src_inputs,
        )

        value_pred = self.critic_head(value_token)

        return value_pred


# TODO: nn.Module?
def init_scheduler(model:nn.Module,shc_cfg):
    raise NotImplementedError()




