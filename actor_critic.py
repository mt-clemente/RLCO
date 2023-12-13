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



    def get_policy(self, state_tokens,segment_tokens, valid_action_mask):


        policy_pred = self.actor.forward(
                tgt_inputs=state_tokens,
                src_inputs=segment_tokens,
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
            nn.Tanh(),
            # nn.Linear(dim_embed,dim_embed,device=device),
            # nn.Tanh(),
            # nn.Linear(cfg['hidden_size'],dim_embed,device=device),
        )
    

        if self.ptrnet:
            self.policy_attn_head = Pointer(
                dim_embed,
                device,
                unit,
                )

        else:
            self.actor_head = nn.Sequential(
                nn.Tanh(),
                nn.Linear(dim_embed,num_segments,device=device,dtype=unit),
                nn.BatchNorm1d(num_segments,device=device)
            )

        self.policy_head = MaskedStableSoftmax()


    
    def forward(self,tgt_inputs,src_inputs,invalid_action_mask):
        
        policy_token = self.mlp(
            tgt_inputs
        )

        if self.ptrnet:
            
            policy_logits = self.policy_attn_head(
                memory=src_inputs,
                target=policy_token,
                memory_mask=invalid_action_mask
            )

        else:
            policy_logits = self.actor_head(policy_token)

        policy_pred = self.policy_head(policy_logits,invalid_action_mask)

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



    def forward(self,src_inputs,tgt_inputs,timesteps,tgt_mask=None,src_key_padding_mask=None):
        
        batch_size = tgt_inputs.size(0)

        if tgt_mask is None:
            tgt_mask = torch.triu(torch.ones(tgt_inputs.size()[1], tgt_inputs.size()[1],device=tgt_inputs.device,dtype=bool), diagonal=1)
            # Convert the mask to a boolean tensor with 'True' values below the diagonal and 'False' values on and above the diagonal
            tgt_mask = tgt_mask.bool()


        tgt_tokens = self.transformer(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_mask=tgt_mask,
        )

        idx = torch.arange(batch_size,device=tgt_tokens.device)
        if timesteps.dim() != 1:
            idx.unsqueeze_(-1)

        tgt_tokens = tgt_tokens[idx,timesteps.squeeze()]

        value_pred = self.critic_head(tgt_tokens)
        return value_pred.squeeze()

# TODO: nn.Module?
def init_scheduler(model:nn.Module,shc_cfg):
    raise NotImplementedError()




