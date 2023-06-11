import torch




class Buffer():

    #TODO: custom state init, if some segments of the solution are known

    def __init__(self,cfg,max_ep_len,device='cpu',init_state=None) -> None:
        self.unit = cfg['unit']
        self.capacity = cfg['training']['horizon']
        self.device = device
        self.dim_token=cfg['dim_token']
        self.init_state = init_state
        self.max_ep_len = max_ep_len

    def push(
            self,
            state,
            action,
            policy,
            reward,
            ep_step,
            final
            ):

        self.state_buf[self.ptr+1] = state
        self.policy_buf[self.ptr] = policy
        self.rew_buf[self.ptr] = reward
        self.timestep_buf[self.ptr] = ep_step
        self.final_buf[self.ptr] = final
        self.act_buf[self.ptr] = action

        self.ptr += 1

    def reset(self):
        self.state_buf = torch.zeros((self.capacity,self.max_ep_len,self.dim_token),device=self.device,dtype=self.unit)
        if not self.init_state is None:
            self.state_buf[0] = self.init_state
        self.act_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.rtg_buf = torch.zeros((self.capacity),device=self.device,dtype=self.unit)
        self.policy_buf = torch.empty((self.capacity),device=self.device,dtype=self.unit)
        self.mask_buf = torch.empty((self.capacity,self.capacity),dtype=bool,device=self.device)
        self.rew_buf = torch.empty((self.capacity),device=self.device,dtype=self.unit)
        self.final_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.adv_buf = torch.zeros((self.capacity),device=self.device,dtype=self.unit)
        self.timestep_buf = torch.empty((self.capacity),device=self.device,dtype=int)
        self.ptr = 0


