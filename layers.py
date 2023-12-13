from einops import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):

    #TODO: Cache encoder output

    def __init__(
            self,
            d_model,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            nhead,
            activation,
            device,
            batch_first,
            norm_first,
            dtype=torch.float,
            return_mem=True,
            dropout=0
            ) -> None:

        super().__init__()

        self.transformer = nn.Transformer(
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype
        )
        
        self.return_mem = return_mem


    def forward(self,src,tgt,tgt_mask,src_key_padding_mask=None,tgt_key_padding_mask=None):


        memory = self.transformer.encoder(src,src_key_padding_mask=src_key_padding_mask)

        if tgt_key_padding_mask is None:
            output = self.transformer.decoder(tgt,memory,tgt_mask=tgt_mask*(-1e6), memory_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask*(-1e6),
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask*-1e6
                )

        if self.return_mem:
            return output,memory
        
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, seq, orig_ch = tensor.shape
        x = int(seq**0.5)
        y = int(seq**0.5)
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc.view(tensor.size())
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
    

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, max_len=5000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param max_len: The maximum length of the sequence.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels

        # Create a long enough 'pe' matrix that can be sliced for shorter sequences
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-np.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register 'pe' as a buffer that does not require gradients
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: A 3d tensor of size (batch_size, sequence_length, channels)
        """
        return self.pe[:x.size(1)].to(x.device)



class Pointer(nn.Module):

    def __init__(self, d_model, device, unit):
        super().__init__()
        self.d_model = d_model

        self.Wq = nn.Linear(d_model,d_model, device=device, dtype=unit)
        self.Wk = nn.Linear(d_model,d_model, device=device, dtype=unit)

    def forward(self, memory:torch.Tensor, target:torch.Tensor, memory_mask:torch.BoolTensor):
        q = self.Wq(target)
        k = self.Wk(memory)
        out = einsum(q,k,'b t, b s t -> b s').squeeze()
        probs = F.softmax(out - 1e9 * memory_mask,dim=-1)
        return probs
    


class MaskedStableSoftmax(nn.Module):
    def __init__(self, eps = 1e8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits,mask):

        # print(torch.logical_not(mask).count_nonzero(dim=-1))
        # print(torch.argmax(torch.softmax(logits - self.eps*mask,-1),-1))
        if 0 in torch.logical_not(mask).count_nonzero(-1):
            raise Exception("No playable tile")
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return torch.softmax(logits - self.eps*mask,-1) 
