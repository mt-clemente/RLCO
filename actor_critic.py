import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MaskedStableSoftmax, Pointer, PositionalEncoding, PositionalEncoding2D, Transformer

class ActorCritic(nn.Module):

    """
    Actor critic model.

    TODO:
    -----------------
    FIXME: UNIT MGT
    Add custom weight init
    """

    def __init__(self, cfg, num_segments,init_state=None, device='cpu'):
        super().__init__()

        self.num_segments = num_segments

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

        self.positional_encoding = PositionalEncoding2D(cfg['dim_embed']) # FIXME: choose type
        self.embed_state_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])
        self.embed_segment_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])

        if cfg['separate_value_training']:
            act_opt_cfg = cfg['actor']['optimizer']
            self.actor_optimizer = init_optimizer(self,opt_cfg=act_opt_cfg)
            crt_opt_cfg = cfg['actor']['optimizer']
            self.critic_optimizer = init_optimizer(self,opt_cfg=crt_opt_cfg)

        else:
            self.optimizer = init_optimizer(self,opt_cfg=cfg['optimizer'])


    def make_transformer_inputs(self,embedded_states,embedded_segments, timesteps):

        
        batch_size = embedded_states.shape[0]

        # FIXME: add positional encoding
        # embedded_states += self.positional_encoding(embedded_states)

        src_inputs = self.embed_segment_ln(embedded_segments)
        tgt_inputs = self.embed_state_ln(embedded_states)
        tgt_key_padding_mask = torch.arange(self.num_segments+1,device=timesteps.device).repeat(batch_size,1) > timesteps

        return src_inputs, tgt_inputs, tgt_key_padding_mask



    def get_policy(self, state_tokens, segment_tokens, timesteps, valid_action_mask,src_key_padding_masks,tgt_mask):

        timesteps = timesteps.unsqueeze(-1)
        src_inputs, tgt_inputs, tgt_key_padding_mask = self.make_transformer_inputs( #FIXME: homogeneous output size 
            embedded_states=state_tokens,
            embedded_segments=segment_tokens,
            timesteps=timesteps
        )

        policy_pred = self.actor.forward(
                src_inputs=src_inputs,
                tgt_inputs=tgt_inputs,
                timesteps=timesteps,
                valid_action_mask=valid_action_mask,
                src_key_padding_mask=src_key_padding_masks,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
        )

        return policy_pred

    def get_value(self, state_tokens, segment_tokens, timesteps):


        batch_size = state_tokens.shape[0]
        timesteps_ = timesteps.unsqueeze(-1)

        embedded_segments = self.embed_segment(segment_tokens)
        embedded_states = self.embed_state(state_tokens)

        embedded_states += self.positional_encoding(embedded_states)

        src_inputs = self.embed_segment_ln(embedded_segments)
        tgt_inputs = self.embed_state_ln(embedded_states)

        tgt_key_padding_mask = torch.arange(self.num_segments+1,device=timesteps_.device).repeat(batch_size,1) > timesteps_

        value = self.critic(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_key_padding_mask=tgt_key_padding_mask
        )

        return value

    
    def get_action(self, policy:torch.Tensor):
        return torch.multinomial(policy,1)
    




class Actor(nn.Module):


    def __init__(self, cfg, dim_embed, num_segments, device, unit) -> None:
        super().__init__()

        self.num_segments = num_segments        
        self.ptrnet = cfg['pointer']
        self.dim_embed=dim_embed

        self.transformer =  Transformer(
            d_model=self.dim_embed,
            num_encoder_layers=cfg['n_encoder_layers'],
            num_decoder_layers=cfg['n_decoder_layers'],
            nhead=cfg['nhead'],
            dim_feedforward=cfg['hidden_size'],
            dropout=0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
            return_mem=cfg['pointer'],
            dtype=unit,
            device=device,
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





    
    def forward(self,src_inputs,tgt_inputs,timesteps,valid_action_mask,tgt_mask=None,src_key_padding_mask=None,tgt_key_padding_mask=None):
        
        batch_size = tgt_inputs.size(0)

        # autoregressive by default
        if tgt_mask is None:
            tgt_mask = torch.triu(torch.ones(tgt_inputs.size()[1], tgt_inputs.size()[1],device=tgt_inputs.device), diagonal=1)
            tgt_mask = tgt_mask.bool()

        if self.ptrnet:

            tgt_tokens, mem_tokens = self.transformer(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            tgt_tokens = tgt_tokens[torch.arange(batch_size,device=tgt_tokens.device),timesteps.squeeze(-1)].reshape(batch_size,self.dim_embed)
            mem_tokens = mem_tokens.reshape(batch_size,self.num_segments,self.dim_embed)
            policy_pred = self.policy_attn_head(mem_tokens,tgt_tokens,torch.logical_not(valid_action_mask))
        
        else:
            policy_tokens = self.transformer(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            policy_logits = self.actor_head(policy_tokens[torch.arange(batch_size,device=policy_tokens.device),timesteps].reshape(batch_size,self.dim_embed))
            policy_pred = self.policy_head(policy_logits,valid_action_mask)

        return policy_pred


class Critic(nn.Module):

    def __init__(self, cfg, dim_embed, device, unit) -> None:
        super().__init__()

        self.transformer =  nn.Transformer(
            d_model=dim_embed,
            num_encoder_layers=cfg['n_encoder_layers'],
            num_decoder_layers=cfg['n_decoder_layers'],
            dim_feedforward=cfg['hidden_size'],
            nhead=cfg['nhead'],
            dropout=0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
            dtype=unit,
            device=device,
        )
    
    
        self.critic_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_embed, 1,device=device,dtype=unit),
        )

        self.dim_embed = dim_embed



    def forward(self,src_inputs,tgt_inputs,timesteps,tgt_mask=None,src_key_padding_mask=None,tgt_key_padding_mask=None):
        
        batch_size = tgt_inputs.size(0)

        if tgt_mask is None:
            tgt_mask = torch.triu(torch.ones(tgt_inputs.size()[1], tgt_inputs.size()[1],device=tgt_inputs.device), diagonal=1)
            # Convert the mask to a boolean tensor with 'True' values below the diagonal and 'False' values on and above the diagonal
            tgt_mask = tgt_mask.bool()

        tgt_tokens = self.transformer(
            src=src_inputs,
            tgt=tgt_inputs,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        value_pred = self.critic_head(tgt_tokens[torch.arange(batch_size,device=tgt_tokens.device),timesteps.squeeze()+1].reshape(batch_size,self.dim_embed))

        return value_pred


# TODO: nn.Module?
def init_scheduler(model:nn.Module,shc_cfg):
    raise NotImplementedError()



def init_optimizer(model:nn.Module,opt_cfg):

    if 'type' in opt_cfg.keys():
        match 'type':
            case 'Adam':
                optimizer = torch.optim.Adam

            case 'SGD':
                optimizer = torch.optim.SGD

            case 'RMSProp':
                optimizer = torch.optim.RMSprop
    
    else:
        optimizer = torch.optim.RMSprop

    return optimizer(model.parameters(),**opt_cfg)

