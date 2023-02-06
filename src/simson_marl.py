#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gym import spaces
import numpy as np
import functools
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

import os
import numpy as np
import math
import time
from mpi4py import MPI

#%% Helping classes and functions

class RingBuffer():
    "A n-dimensional ring buffer using numpy arrays"
    def __init__(self, length, dim=1):
        if type(dim) is int:
            dim = (dim,)
        self.data = np.zeros((length,)+dim, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        assert x.shape==self.data.shape[1:],'Input array does not match \
            the ring buffer size'
        # breakpoint()
        # x_index = (self.index + np.arange(x.size)) % self.data.size
        x_index = self.index % self.data.shape[0]
        self.data[x_index] = x
        self.index = x_index + 1#[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

    def average(self):
        "Returns the average of the entries in the ring buffer"
        return np.mean(self.data,axis=0)

#%% PettingZoo environment

def env():
    """
    The env function often wraps the environment in wrappers by default.
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(conf, rank_folder):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(conf, rank_folder)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, conf, rank_folder):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        
        self.conf = conf

        # Depending on the wall-boundary condition number,
        # different control problems are implemented
        #   - wbci = 1: Fixed amplitude, localized in x direction with
        #               smooth step function. Tunable frequency of the
        #               spanwise oscillation (NOT IMPLEMENTED)
        #   - wbci = 6: Opposition control, tunable amplitude
        #   - wbci = 7: Localized control, for multi-agent RL 
        if ((self.conf.simulation.wbci == 6) or (self.conf.simulation.wbci == 7)):
            self.ctrl_min_amp = self.conf.runner.ctrl_min_amp
            self.ctrl_max_amp = self.conf.runner.ctrl_max_amp
            self.action_shape = [1,]#[self.conf.simulation.nctrlz,]
        else:
            raise ValueError("This wall boundary condition option is not available")

        # List of possible agents
        self.possible_agents = [f"jet_z{r}_x{s}" for s in range(self.conf.simulation.nctrlx) 
                                                 for r in range(self.conf.simulation.nctrlz)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Defining the size of the observations, based on the number of agents
        assert (self.conf.simulation.nzjet-1) <= self.conf.simulation.nctrlz,\
               f"Too large nzjet value ({self.conf.simulation.nzjet})"
        assert (self.conf.simulation.nxjet-1) <= self.conf.simulation.nctrlx,\
               f"Too large nxjet value ({self.conf.simulation.nxjet})"
        
        if self.conf.simulation.nctrlz>1:
            if self.conf.simulation.nctrlz==self.conf.simulation.nz:
                # if self.conf.runner.partial_reward==True:
                print("[W] When the number of agents is the same as the number of points in "+\
                        "the simulations, the observation cannot include the neighbouring jets")
                self.conf.simulation.nzjet = 1
                self.nzs_ = 1
                self.conf.simulation.nzs = self.nzs_
                # else:
                #     self.conf.simulation.nzs = self.conf.simulation.nz
            else:
                # Each agent can see the field portion in the spanwise direction
                # that includes also the neightbouring jets...
                self.nzs_ = self.conf.simulation.nzjet*\
                    self.conf.simulation.nz//self.conf.simulation.nctrlz
                self.conf.simulation.nzs = self.nzs_+\
                        int(self.conf.simulation.zchange)+1
            
        else:
            self.conf.simulation.nzs = self.conf.simulation.nz

        if self.conf.simulation.nctrlx>1:
            if self.conf.simulation.nctrlx==self.conf.simulation.nx:
                # if self.conf.runner.partial_reward==True:
                print("[W] When the number of agents is the same as the number of points in "+\
                    "the simulations, the observation cannot include the neighbouring jets") 
                self.conf.simulation.nxjet = 1
                self.nxs_ = 1
                self.conf.simulation.nxs = self.nxs_
                # else:
                #     self.conf.simulation.nxs = self.conf.simulation.nx
            else:
                # ...and similarly in the streamwise direction
                self.nxs_ = self.conf.simulation.nxjet*\
                    self.conf.simulation.nx//self.conf.simulation.nctrlx
                self.conf.simulation.nxs = self.nxs_+\
                    int(self.conf.simulation.xchange)+1
        else:
            self.conf.simulation.nxs = self.conf.simulation.nx
        # if e.g. nzjet=3, one neighbouring jet per side is seen by the agent

        # Launch variables
        self.folder = rank_folder
        self.restart_index = 0
        
        # Reward history logging variables
        self.act_index = 0
        self.reward_log = list()
        
        # Action rescaling variables
        self.rescale_actions = False
        self.rescale_factors = list()

        # Derived simulation variables
        self.dy = 1-np.cos(math.pi*1/(self.conf.simulation.ny-1))
        
        if ((self.conf.simulation.wbci == 6) or (self.conf.simulation.wbci == 7)):
            # Load the reference average du/dy at the wall to use as baseline
            baseline_dudy_dict = {"180_16x65x16"   : 3.7398798426242075,
                                  "180_32x33x32"   : 3.909412638928125,
                                  "180_32x65x32"   : 3.7350180468974763,# 
                                  "180_64x65x64"   : 3.82829465265046,
                                  "180_128x65x128" : 3.82829465265046}
            self.baseline_dudy = baseline_dudy_dict[f"{int(conf.simulation.retau)}_" + \
                                f"{conf.simulation.nx}x{conf.simulation.ny}x{conf.simulation.nz}"]
        
        # State-related variables
        if self.conf.simulation.npl-1 > self.conf.runner.npl_state:
            self.full_observation = \
                np.ndarray((self.conf.simulation.npl-1,
                            self.conf.simulation.nz, 
                            self.conf.simulation.nx))

        # Reward-related variables
        if self.conf.runner.rew_mode == 'MovingAverage':
            self.reward_history = RingBuffer(length=self.conf.runner.size_history,
                                            dim=(self.conf.simulation.nzs, 
                                                 self.conf.simulation.nxs))
            # The history of the wall shear-stress is initialized with
            # an the reference value
            for i_h in range(self.conf.runner.size_history):
                self.reward_history.data[i_h] = self.baseline_dudy*np.ones((conf.simulation.nzs,
                                                                            conf.simulation.nxs))

        # Input scaling options
        # TODO load statistical values to normalize the observations
        #if self.conf.runner.normalize_input == True:
        #    pass
        
        # Spawning SIMSON as MPI process
        mpi_info = MPI.Info.Create()
        mpi_info.Set('wdir',f"{os.getcwd()}/{self.folder}")
        mpi_info.Set('bind_to','none')
        if conf.simulation.hostfile != '':
            mpi_info.Set('hostfile',conf.simulation.hostfile)
        self.mpi_info = mpi_info
        self.sub_comm = MPI.COMM_SELF.Spawn(f'./bla_{conf.simulation.nx}x{conf.simulation.ny}x{conf.simulation.nz}_{conf.simulation.nproc}', 
                                    maxprocs=self.conf.simulation.nproc,
                                    info=self.mpi_info)

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low = self.ctrl_min_amp,
                          high = self.ctrl_max_amp,
                          shape = self.action_shape,
                          dtype = np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low = -np.inf,
                          high = np.inf,
                          shape = (self.conf.runner.npl_state,
                                  self.conf.simulation.nzs,
                                  self.conf.simulation.nxs),
                                  dtype = np.float32)


    def render(self, mode='human', close=False):
        ...

    def close(self):
        self.end_simulation()
        # Wait until all the operations are completed
        time.sleep(10)

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]

        # Close the current SIMSON simulation
        self.end_simulation()
        
        # Save reward log and re-initialized it
        np.savez(self.folder+f'/rewlog_{self.restart_index}.npz',
                rew=np.array(self.reward_log))
        self.reward_log = list()
        
        # Rename old output file from SIMSON
        try:
            os.rename(self.folder+'/main.out',self.folder+f'/main_{self.restart_index}.out')
            self.restart_index+=1
            print('File renaming successful')
        except FileNotFoundError:
            # In order to have anyway the correct value for the reward log
            self.restart_index+=1
            pass
        
        # Re-initialize the action index
        self.act_index = 0

        if self.conf.runner.rew_mode == 'MovingAverage':
            # The history of the wall shear-stress is re-initialized with
            # the same existing value used in __init__()
            for i_h in range(self.conf.runner.size_history):
                self.reward_history.data[i_h] = self.baseline_dudy*np.ones((self.conf.simulation.nz,
                                                                            self.conf.simulation.nx))
        
        # Change initial field is necessary
        if self.conf.runner.random_init > 0:
            import re
            n_init = np.random.randint(self.conf.runner.random_init)
            with open(self.folder+'/bla.i','r+') as f:
                lines = f.readlines()
                line = re.sub(r"_0..\.u",f"_{n_init:03d}.u",lines[3])
                lines[3] = line
                f.seek(0)
                f.writelines(lines)

        # Open a new simulation
        self.start_simulation()
        
        # Return current (initial) state
        time, observation = self.state()

        # Distribute observations to the agents
        observations = self._distribute_field(observation,reward=False)

        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # Resetting reward value
        rewards = {}

        # Trasform actions dict into an array
        if self.conf.simulation.wbci == 6 or self.conf.simulation.wbci == 7:
            ctrl_value = np.zeros((self.conf.simulation.nctrlz,self.conf.simulation.nctrlx))            
            for i_a,agent in enumerate(self.agents):
              i_z=i_a%self.conf.simulation.nctrlz
              i_x=i_a//self.conf.simulation.nctrlz
              ctrl_value[i_z,i_x] = actions[agent]  
        # if ((self.conf.simulation.wbci == 6) or (self.conf.simulation.wbci == 7)):
        #     
        # for i in range(self.conf.simulation.nctrlz):
        #     ctrl_value[i] = action[i]

        # Linear mapping of the actions if the range differs
        # breakpoint()
        if self.rescale_actions:
            for i_a,agent in enumerate(self.agents):
                if actions[agent] < 0:
                    actions[agent] *= self.rescale_factors[0][0]
                if actions[agent] > 0:
                    actions[agent] *= self.rescale_factors[0][1] 

        # Sending the new action values to the environment
        self.action(ctrl_value)

        # Let the solution evolve with the new control values
        rewards = self.evolve()
        
        # Obtain new observation
        time, observation = self.state()

        # Distribute observations to the agents
        observations = self._distribute_field(observation,reward=False)

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]['time'] = time
        # Check whether we have approximately reached the maximum simulation time
        if time > self.conf.simulation.tmax - 1:
            dones = {agent : True for agent in self.agents}
        else:
            dones = {agent : False for agent in self.agents}
        # Add check regarding the maximum number of interactions
        if self.act_index >= self.conf.runner.nb_interactions:
            dones = {agent : True for agent in self.agents}

        return observations,rewards,dones,infos

    def _distribute_field(self,field,reward=True):

        # TODO modify to accept also nx==nctrlx or nz==nctrlz
        # TODO add the possibility to include the neighbouring jets if
        #      nx==nctrlx and/or nz==nctrlz

        # agent_field = np.ndarray((npls,
        #                           self.conf.simulation.nzs,
        #                           self.conf.simulation.nxs),
        #                           dtype=np.double)
        
        distributed_fields = {}

        field = np.roll(field,self.conf.simulation.nx//2,axis=2)

        if self.conf.simulation.nctrlz!=self.conf.simulation.nz \
           and self.conf.simulation.nctrlx!=self.conf.simulation.nx:
            zcenters = np.arange(
                self.conf.simulation.nz//self.conf.simulation.nctrlz//2,
                self.conf.simulation.nz,
                step=self.conf.simulation.nz//self.conf.simulation.nctrlz)

            xcenters = np.arange(
                self.conf.simulation.nx//self.conf.simulation.nctrlx//2,
                self.conf.simulation.nx,
                step=self.conf.simulation.nx//self.conf.simulation.nctrlx)
            
            for i_a,agent in enumerate(self.agents):
                i_z=i_a%self.conf.simulation.nctrlz
                i_x=i_a//self.conf.simulation.nctrlz
                i_zc = i_z#self.conf.simulation.nctrlz-i_z-1
                i_xc = i_x

                if zcenters[i_zc]<self.nzs_//2+int(self.conf.simulation.zchange):
                    zroll = self.nzs_//2+int(self.conf.simulation.zchange)
                elif zcenters[i_zc]+self.nzs_//2+int(self.conf.simulation.zchange)>self.conf.simulation.nz:
                    zroll = -(self.nzs_//2+int(self.conf.simulation.zchange))
                else:
                    zroll=0

                if xcenters[i_xc]<self.nxs_//2+int(self.conf.simulation.xchange):
                    xroll = self.nxs_//2+int(self.conf.simulation.xchange)
                elif xcenters[i_xc]+self.nxs_//2+int(self.conf.simulation.xchange)>self.conf.simulation.nx:
                    xroll = -(self.nxs_//2+int(self.conf.simulation.xchange))
                else:
                    xroll=0

                agent_field = np.roll(field,
                    (zroll,xroll),axis=(1,2))[:,zcenters[i_zc]+
                            zroll-self.nzs_//2:zcenters[i_zc]+
                            zroll+self.nzs_//2+
                            int(self.conf.simulation.zchange)+1,xcenters[i_xc]+
                            xroll-self.nxs_//2:xcenters[i_xc]+
                            xroll+self.nxs_//2+
                            int(self.conf.simulation.xchange)+1]
                
                distributed_fields[agent] = agent_field

        elif self.conf.simulation.nctrlz!=self.conf.simulation.nz \
           or self.conf.simulation.nctrlx!=self.conf.simulation.nx:
            raise NotImplementedError('This feature is not currently available')
        else:
            for i_a,agent in enumerate(self.agents):
                if self.conf.runner.partial_reward or reward!=True:
                    i_z=i_a%self.conf.simulation.nctrlz
                    i_x=i_a//self.conf.simulation.nctrlz
                    distributed_fields[agent] = np.reshape(field[:,i_z,i_x],[-1,1,1])
                else:
                    distributed_fields[agent] = field
        # breakpoint()
        return distributed_fields


#%% Request methods

    def start_simulation(self):
        # Open a new simulation
        self.sub_comm = MPI.COMM_SELF.Spawn(f"{self.mpi_info['wdir']}"+
            f'/bla_{self.conf.simulation.nx}x{self.conf.simulation.ny}x{self.conf.simulation.nz}_{self.conf.simulation.nproc}', 
            maxprocs=self.conf.simulation.nproc,
            info=self.mpi_info)
        
        
    def end_simulation(self):
        request = b'TERMN'
        
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0, 
                        tag=self.conf.simulation.nproc+100)
        
        self.sub_comm.Disconnect()




    def state(self):
        request = b'STATE'
        
        # print("Python sending state message to Fortran")
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0, 
                        tag=self.conf.simulation.nproc+100)
        
        # the reward is assumed to be the first plane requested in bla.i
        # the state, all the remaining
        current_state = np.ndarray((self.conf.simulation.npl-1,
                                    self.conf.simulation.nz,
                                    self.conf.simulation.nx),dtype=np.double)
        buffer = np.ndarray((self.conf.simulation.nx,),dtype=np.double)
        
        for i in range(1,self.conf.simulation.npl):
            for i_z in range(self.conf.simulation.nz):
                self.sub_comm.Recv([buffer, MPI.DOUBLE], 
                                source=0, 
                                tag=self.conf.simulation.nproc+10+i+i_z+1)
                current_state[i-1,i_z] = buffer
                
        # Receiving current time of the simulation
        current_time = np.array(0,dtype=np.double)
        self.sub_comm.Recv([current_time, MPI.DOUBLE], source=0,
                        tag=self.conf.simulation.nproc+960)
        
        # Normalizing the state output (currently only zero-mean) 
        if self.conf.runner.normalize_input != "None":
            for i in range(self.conf.runner.npl_scale):
                current_state[i] = (current_state[i]-np.mean(current_state[i]))
                if self.conf.runner.normalize_input == 'utau':
                    current_state[i] /= self.conf.runner.ctrl_max_amp
                # assuming that plane after npl_state contains the wall-shear
                # stress to compute utau
                elif self.conf.runner.normalize_input == 'current_utau':
                    # compute utau
                    current_utau = np.sqrt(np.mean(current_state[self.conf.runner.npl_state])\
                                   /self.dy/self.conf.simulation.Re_cl)
                    current_state[i] /= current_utau
                    print(current_utau, flush=True)
                elif self.conf.runner.normalize_input == 'std':  
                    current_state[i] /= np.std(current_state[i])
                elif self.conf.runner.normalize_input == 'zero mean':
                    continue
                else:
                    raise ValueError('Unknown input scaling option')

        # np.savez(self.conf.logging.save_dir+f'/{self.conf.logging.run_name}/env_000/obs_state.npz',
        #         ob=current_state)
        # XXX Workaround!
        if self.conf.simulation.nctrlz==self.conf.simulation.nz:
            current_state[0] = np.roll(current_state[0],(-1,),axis=(0,))
            current_state[1] = np.roll(current_state[1],(-1,),axis=(0,))
        if self.conf.simulation.nctrlx==self.conf.simulation.nx:
            current_state[0] = np.roll(current_state[0],(-1,),axis=(1,))
            current_state[1] = np.roll(current_state[1],(-1,),axis=(1,))

        if self.conf.simulation.npl-1 > self.conf.runner.npl_state:
            self.full_observation = np.copy(current_state)
            current_state = current_state[:self.conf.runner.npl_state]

        return current_time, current_state


    def action(self,ctrl_value):
        request = b'CNTRL'
        
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0,
                        tag=self.conf.simulation.nproc+100)
        # breakpoint()
        if ((self.conf.simulation.wbci == 6) or (self.conf.simulation.wbci == 7)):
            for j in range(self.conf.simulation.nctrlx):
                for i in range(self.conf.simulation.nctrlz):
                    self.sub_comm.Send([ctrl_value[i,j], MPI.DOUBLE], 
                                        dest=0,
                                        tag=300+i+1+(j+1)*self.conf.simulation.nctrlz)
        
        
    def evolve(self):
        request = b'EVOLV'
        i_evolv = 1
        
        # print("Python sending evolve message to Fortran")
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0,
                        tag=self.conf.simulation.nproc+100)
        
        uxz = np.ndarray((self.conf.simulation.nz,
                        self.conf.simulation.nx),dtype=np.double)
        
        uxz_avg = np.ndarray((self.conf.simulation.nz,
                        self.conf.simulation.nx),dtype=float)
        
        buffer = np.ndarray((self.conf.simulation.nx,),dtype=np.double)
        
        # 'Receiving the (non-averaged) reward'
        while i_evolv <= (self.conf.simulation.ndrl // self.conf.simulation.nst)-1:
            """
            print(f'Receiving the (non-averaged) reward, i_evolv:{i_evolv}',
                flush=True)
            """
            for i in range(1):
                for i_z in range(self.conf.simulation.nz):
                    self.sub_comm.Recv([buffer, MPI.DOUBLE], 
                                    source=0, 
                                    tag=self.conf.simulation.nproc+10+i+i_z+1)
                    uxz[i_z] = buffer
            
            # Reward averaging (between two consecutive actions)
            if i_evolv == 1:
                uxz_avg = uxz.astype(float)/self.dy
            else:
                uxz_avg = (uxz_avg*i_evolv + uxz.astype(float)/self.dy)/(i_evolv+1)
            
            i_evolv += 1
        
        if self.conf.runner.rew_mode == 'Instantaneous':
            avg_array = uxz_avg#np.mean(uxz_avg,axis=1)
        elif self.conf.runner.rew_mode == 'MovingAverage':
            self.reward_history.extend(uxz_avg)#np.mean(uxz_avg,axis=1))
            avg_array = self.reward_history.average()
        else:
            raise ValueError('Unknown reward computing mode')
        
        # Distributing the field to compute the individual reward per each agent
        ws_stresses = self._distribute_field(np.expand_dims(avg_array,0),reward=True)

        # Computing the reward
        print(f'Actuation: {self.act_index+((self.restart_index-1)*self.conf.runner.nb_interactions)}',flush=True)
        rewards = {}
        for i_a,agent in enumerate(self.agents):
            if ((self.conf.simulation.wbci == 6) or (self.conf.simulation.wbci == 7)):
                reward = 1-np.mean(ws_stresses[agent])/self.baseline_dudy
            else:
                raise ValueError('Unknown boundary condition option')
            # TODO add a baseline value
            rewards[agent] = reward
            if self.conf.simulation.nctrlz!=self.conf.simulation.nz \
               and self.conf.simulation.nctrlx!=self.conf.simulation.nx:
                print(f"Reward {i_a}: {reward}",flush=True)
            elif i_a == 0:
                print(f"Reward {i_a}: {reward}",flush=True)
        print("",flush=True)
        
        # Logging reward for debugging and further analysis
        self.reward_log.append(rewards)
        # if self.conf.runner.rew_mode == 'MovingAverage':
        # np.savez(self.folder+f'/act{self.plot_index}.npz',rew=avg_array,
        #          inst_rew=uxz_avg[0,0])
        self.act_index += 1
        
        return rewards
