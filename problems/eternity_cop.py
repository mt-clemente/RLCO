from pathlib import Path
import torch
from cop_class import COPInstanceBatch, COProblem, COPInstance
from eternity.eternity import EternityPuzzle
from eternity.utils import initialize_sol
from torch import nn
from einops import rearrange


class Eternity(COProblem):

    def __init__(self, dim_embedding, device=None) -> None:
        super().__init__(dim_embedding, device)

        assert not dim_embedding % 4 

        MAX_COLORS = 23
        self.color_embedding_size = dim_embedding//4


        self.color_embedding = nn.Embedding(
            num_embeddings=MAX_COLORS,
            embedding_dim=self.color_embedding_size,
            device=device
            )

    def load_instance(self, path: Path) -> COPInstance:
        pz = EternityPuzzle(path)
        state, tiles, _, n_tiles = initialize_sol(pz,self.device)
        self.num_segments = n_tiles
        return COPInstance(state,tiles,size = int(n_tiles**0.5))
    

    # TODO: Fix kwargs
    def act(self, states:torch.Tensor, segments:torch.Tensor, steps:torch.IntTensor, **kwargs) -> tuple:

        rewards = torch.zeros(len(states),device=states.device)
        new_states = torch.empty_like(states)

        for i, (state, segment, step, size) in enumerate(zip(
            states,
            segments,
            steps,
            kwargs['size']
            )):


            n_state, reward = self.place_tile(state,segment,size,step)

            new_states[i] = n_state
            rewards[i] = reward

        return new_states, rewards
    
    def tokenize(self, instance:COPInstance) -> tuple[torch.Tensor,torch.Tensor]:

        state_ = self.color_embedding(instance.state)
        segments_ = self.color_embedding(instance.segments)

        state_tokens = rearrange(state_, "b h w e -> (b h w) e")
        segment_tokens = rearrange(segments_, "b h w e -> b (h w) e")

        return state_tokens, segment_tokens
    
    def valid_action_mask(self, state) -> torch.BoolTensor:
        # TODO:
        return super().valid_action_mask(state)


    # GAME UTILS

    def place_tile(self,state:torch.Tensor,tile:torch.Tensor,ep_step:int,size:int,step_offset:int=0):
        """
        If you start with a prefilled board with k pieces, you need to place tiles at spot
        k + 1, hence the need for a step offset.

        TODO: Cleanup
        """
        step = ep_step + step_offset
        state = state.clone()
        # size = state.size()[0] - 2
        best_conflict = -10
        best_connect = -1
        best_state = None
        best_reward = None

        for _ in range(4):
            tile = tile.roll(self.color_embedding_size,-1)
            state[step // size + 1, step % size + 1,:] = tile
            conflicts, connect, reward = self.filling_connections(state,size,step)

            if connect > best_connect:
                best_state=state.clone()
                best_connect = connect
                best_conflict = conflicts
                best_reward = reward


        return best_state, best_conflict



    def filling_connections(self,state:torch.Tensor, bsize:int, step):
        """
        Calculates the created conflicts and connections when placing one tile
        """
        i = step // bsize + 1
        j = step % bsize + 1
        west_tile_color = state[i,j-1,3*self.color_embedding_size:4*self.color_embedding_size]
        south_tile_color = state[i-1,j,:self.color_embedding_size]

        west_border_color = state[i,j,1*self.color_embedding_size:2*self.color_embedding_size]
        south_border_color = state[i,j,2*self.color_embedding_size:3*self.color_embedding_size]

        sides = 0
        connections = 0
        reward = 0

        sides += 1
        if j == 1:
            if torch.all(west_border_color == 0):
                reward += 2
                connections += 1
        
        elif torch.all(west_border_color == west_tile_color):
            connections += 1
            reward += 1

        sides += 1
        if i == 1:
            if torch.all(south_border_color == 0):
                connections += 1
                reward += 2
        
        elif torch.all(south_border_color == south_tile_color):
            connections += 1
            reward += 1
    
    
        if j == bsize:

            east_border_color = state[i,j,3*self.color_embedding_size:4*self.color_embedding_size]
            sides += 1
            if torch.all(east_border_color == 0):
                connections += 1
                reward += 2


        if i == bsize:

            north_border_color = state[i,j,:self.color_embedding_size]
            sides += 1
            if torch.all(north_border_color == 0):
                reward += 2
                connections += 1
        
        return sides - connections, connections, reward

