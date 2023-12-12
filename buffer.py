import torch




class Buffer():

    #TODO: custom state init, if some segments of the solution are known
    # CHECK WHICH ELEMNTS HAVE TO BE BUFFERED AND WHICH DO NOT ---> HORIZON VS SEQUENCE

    def __init__(self,cfg,num_instances,max_ep_len,dim_token,max_num_segments,device='cpu',init_state=None) -> None:
        self.unit = cfg['network']['unit']
        self.capacity = cfg['training']['horizon']
        self.device = device
        self.dim_token=dim_token
        self.max_num_segments = max_num_segments
        self.init_state = init_state
        self.max_ep_len = max_ep_len
        self.num_instances = num_instances
        self.reset()
        self.reset_sol()

    def push(
            self,
            state,
            action,
            policy,
            reward,
            mask,
            ep_step,
            final
            ):

        self.state_buf[:,self.ptr] = state
        self.solution_buf[:,self.sol_ptr] = state
        self.policy_buf[:,self.ptr] = policy
        self.mask_buf[:,self.ptr] = mask
        self.rew_buf[:,self.ptr] = reward
        self.timestep_buf[:,self.ptr] = ep_step
        self.final_buf[:,self.ptr] = final
        self.act_buf[:,self.ptr] = action
        self.ptr += 1
        self.sol_ptr += 1


    def push_horizon(
            self,
            state,
            step
            ):
        
        self.horzion_states = state
        self.horzion_timesteps = step

    def reset(self):
        
        self.state_buf = torch.zeros((self.num_instances,self.capacity,self.dim_token),device=self.device,dtype=self.unit) -10000
        if not self.init_state is None:
            self.state_buf[0] = self.init_state
        
        self.value_buf = torch.zeros((self.num_instances,self.capacity),device=self.device,dtype=self.unit)
        self.horzion_timesteps = torch.empty(self.num_instances,device=self.device,dtype=int)
        self.horzion_states = torch.empty((self.num_instances,self.dim_token),device=self.device,dtype=self.unit)
        self.act_buf = torch.empty((self.num_instances,self.capacity),dtype=int,device=self.device)
        self.policy_buf = torch.empty((self.num_instances,self.capacity),device=self.device,dtype=self.unit)
        self.mask_buf = torch.empty((self.num_instances,self.capacity,self.max_num_segments),dtype=bool,device=self.device)
        self.rew_buf = torch.empty((self.num_instances,self.capacity),device=self.device,dtype=self.unit)
        self.final_buf = torch.empty((self.num_instances,self.capacity),dtype=int,device=self.device)
        self.timestep_buf = torch.empty((self.num_instances,self.capacity),device=self.device,dtype=int)
        self.ptr = 0

    def reset_sol(self):
        self.solution_buf = torch.zeros((self.num_instances,self.max_ep_len+1,self.dim_token),device=self.device,dtype=self.unit) -10000
        self.sol_ptr = 0