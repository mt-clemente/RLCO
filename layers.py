import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):

    #TODO: Cache encoder output

    def __init__(self, d_model,num_encoder_layers,num_decoder_layers,dim_feedforward,nhead,activation,device,batch_first,norm_first,dtype=torch.float,return_mem=True,dropout=0) -> None:
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
                dtype=dtype,
        )
        self.return_mem = return_mem


    def forward(self,src,tgt,tgt_mask,src_key_padding_mask=None,tgt_key_padding_mask=None):


        memory = self.transformer.encoder(src,src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt,memory,tgt_mask=tgt_mask.half()*(-1e6), memory_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask.half()*-1e6)

        if self.return_mem:
            return output, memory
        
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
    
class Pointer(nn.Module):

    def __init__(self, d_model, device, unit):
        super().__init__()
        self.d_model = d_model

        self.Wq = nn.Linear(d_model,d_model, device=device, dtype=unit,bias=False)
        self.Wk = nn.Linear(d_model,d_model, device=device, dtype=unit,bias=False)
        self.v = nn.Linear(d_model, 1, device=device, dtype=unit,bias=False)

        torch.nn.init.normal_(self.v.weight,0,0.01) #TODO: add paper stating 0.01 init is good


    def forward(self, memory:torch.Tensor, target:torch.Tensor, memory_mask:torch.BoolTensor):
        q = self.Wq(target).unsqueeze(1)
        k = self.Wk(memory)
        out = self.v(torch.tanh(q + k)).squeeze(-1)
        probs = F.softmax(out - 1e9 * memory_mask,dim=-1)
        return probs
    


class MaskedStableSoftmax(nn.Module):
    def __init__(self, eps = 1e5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits,mask):
        if mask.count_nonzero() == 0:
            raise Exception("No playable tile")
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return torch.softmax(logits - self.eps* torch.logical_not(mask),-1) 
