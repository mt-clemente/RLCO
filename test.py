import torch

a = torch.tensor([[1,2],[1,2],[1,2],[1,2],[1,2]])
b = torch.tensor([[0,1],[1,2],[2,3],[0,2],[1,4]])

dist = torch.sqrt(((a - b)**2).sum(-1))
print(dist)