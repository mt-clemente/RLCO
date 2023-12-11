from pathlib import Path
import torch
from cop_class import COProblem, COPInstance
from matplotlib import pyplot as plt


class TSP(COProblem):

    def __init__(self,cfg):
        env_size =cfg['env_size']
        stops =cfg['stops']
        self.ENV_SIZE = env_size
        self.N_STOPS = stops
        self.maxSteps = stops    
        self.states_size, self.actions_size = stops, stops
        self.token_size = 2
        self.reuse_segments = False

    def load_instance(self, path: Path) -> COPInstance:
        torch.manual_seed(0)
        cities = torch.rand(self.N_STOPS, 2)*self.ENV_SIZE
        cities = torch.hstack((cities,torch.arange(cities.size(0)).unsqueeze(-1)))
        state = torch.zeros_like(cities)
        state[0] = cities[0]
        cities = cities[1:]
        return COPInstance(state,cities,self.N_STOPS,self.N_STOPS-1)


    
    def act(self, states:torch.Tensor, segments:torch.Tensor, steps:torch.IntTensor,sizes) -> tuple:

        n_states = states.clone()
        idx = torch.arange(n_states.size(0),device = states.device)
        n_states[idx,steps+1] = segments[idx]
        rewards = -self.dist(n_states[idx,steps],n_states[idx,steps+1])
        return n_states, rewards

    def to_tokens(self, states, segments):
        return states,segments
    
    def dist(self,s1,s2):
        return torch.sqrt(((s1 - s2)**2)).sum(-1)
    
    def reset(self):
        """Restart the environment for experience replay
        Returns the first state
        """
        self.currentState = 0
        self.stepCount = 0
        self.stops[:, 3] = torch.zeros(self.N_STOPS)
        self.stops[0, 3] = 1    # Visited
        validActions = torch.arange(1, self.N_STOPS)
        return 0, validActions


    def render(self):
        """Visualize the environment state"""
        plt.scatter(self.stops[:, 0], self.stops[:, 1], c='purple')
        plt.axis('off')


    def get_loss(self,states:torch.Tensor,**kwargs):

        shifted = states.roll(1,0)
        return self.dist(states,shifted).sum()

