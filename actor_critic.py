from matplotlib import pyplot as plt
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

        self.dim_embed = cfg['dim_embed'] 

        if 'categorical_vocab_size' in cfg.keys():
            self.embed_state = nn.Embedding(cfg['categorical_vocab_size'],self.dim_embed)
            self.embed_segment = nn.Embedding(cfg['categorical_vocab_size'],self.dim_embed)

        if cfg['pos_encoding'] == 1:
            self.positional_encoding = PositionalEncoding1D(cfg['dim_embed']) # FIXME: choose type
        elif cfg['pos_encoding'] == 2:
            self.positional_encoding = PositionalEncoding2D(cfg['dim_embed']) # FIXME: choose type
        else:
            self.positional_encoding = lambda x:0


        self.critic = Critic(
            cfg=crt_cfg,
            positional_encoding=self.positional_encoding,
            dim_embed=cfg['dim_embed'],
            device=device,
            unit=cfg['unit']
        )



            
        self.embed_state_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])
        self.embed_segment_ln = nn.LayerNorm(self.dim_embed,eps=1e-5,device=device,dtype=cfg['unit'])


        self.causal_mask = torch.triu(torch.ones(max_inst_size, max_inst_size,device=self.device,dtype=bool), diagonal=1)


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

        layers = [nn.Tanh()]
        for i in range(cfg['layers']):
            layers.append(nn.LazyLinear(cfg['hidden_size'],device=device))
            layers.append(nn.GELU())

        self.mlp =  nn.Sequential(
            *layers
        )
    

        if self.ptrnet:
            self.policy_attn_head = Pointer(
                dim_embed,
                device,
                unit,
                )

        else:
            self.actor_head = nn.Sequential(
                nn.LazyLinear(num_segments,device=device,dtype=unit),
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

    def __init__(self, cfg, dim_embed,positional_encoding, device, unit) -> None:
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_embed,
                nhead=cfg['nhead'],
                dim_feedforward=cfg['hidden_size'],
                dropout=0,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                dtype=unit,
                device=device
            ),
            num_layers=cfg['n_encoder_layers']
        )

        self.pos_encoding = positional_encoding
    
    
        self.critic_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_embed, 1,device=device,dtype=unit),
        )

        self.dim_embed = dim_embed



    def forward(self,src_inputs,timesteps,src_key_padding_mask=None):
        # FIXME: 0 at the end of timesteps???
        batch_size = src_inputs.size(0)

        # src_inputs += self.pos_encoding(src_inputs)

        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)

        value_pred = torch.empty_like(timesteps,dtype=src_inputs.dtype)

        if src_inputs.dim() == 4:
            for i in range(timesteps.size(-1)):
                
                step_inputs = src_inputs[:,i]
                src_key_padding_mask = torch.arange(step_inputs.size(1),device=timesteps.device).expand(batch_size,-1) > timesteps[:,i].unsqueeze(-1)

                tokens = self.transformer(
                        src=step_inputs,
                        src_key_padding_mask=src_key_padding_mask
                )

                tokens = tokens[:,0] # TODO: add pooling + concat ===> feed into head
                value_pred[:,i] = self.critic_head(tokens).squeeze()

        else:
                src_key_padding_mask = torch.arange(src_inputs.size(1),device=timesteps.device) > timesteps
                tokens = self.transformer(
                        src=src_inputs,
                        src_key_padding_mask=src_key_padding_mask
                )

                tokens = tokens[:,0] # TODO: add pooling + concat ===> feed into head
                value_pred = self.critic_head(tokens).squeeze()


        return value_pred.squeeze()

# TODO: nn.Module?
def init_scheduler(model:nn.Module,shc_cfg):
    raise NotImplementedError()




