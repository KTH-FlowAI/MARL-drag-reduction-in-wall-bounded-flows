#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from bdb import Breakpoint
from pathlib import Path
from typing import List

import supersuit as ss

import os
import shutil
import sys

import numpy as np
import yaml
from omegaconf import OmegaConf
import math
import matplotlib.pyplot as plt

import json
import time

from configuration import Config
from simsonutils3D import write_simson_input_file
import simson_marl as SimsonEnv

os.environ["CUDA_VISIBLE_DEVICES"]=""

def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("evaluate", help="Run SIMSON")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=evaluate)

def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf

def evaluate(conf_file, overrides, **ignored_kwargs):
    
    # Read configuration from yaml and optional overrides
    conf = parse_omegaconf(conf_file, overrides)
    
    # Check if we are evaluating a previously-trained model
    if conf.runner.agent_run_name != 0:
        conf.logging.run_name = conf.runner.agent_run_name
        conf.runner.load_agent = True
    else:
        # XXX Is this needed?
        conf.runner.rewrite_input_files = True
    
    # Create run folder
    run_folder = conf.logging.save_dir+f'/{conf.logging.run_name}'
    
    if not os.path.exists(run_folder):
        if conf.runner.agent_run_name != 0:
            raise ValueError("The folder containing the trained agent "+\
                              "does not exist")
        os.mkdir(run_folder)

    if conf.runner.load_agent == False:
        pass
        ...
        
    if not conf.runner.evaluation:
        # Write bla.i file in the bin_root folder
        simson_conf_file = write_simson_input_file(conf)

        # make copies of the code to avoid collisions
        print(f"copy binaries for env of rank {conf.runner.rank}",flush=True)

        rank_folder = run_folder+f'/env_{conf.runner.rank:03d}'
        # make the env folder and copy all the necessary files
        if not os.path.exists(rank_folder):
            os.mkdir(rank_folder)
        
        shutil.copy(conf.runner.bin_root + \
            f"/bla_{conf.simulation.nx}x{conf.simulation.ny}x{conf.simulation.nz}_{conf.simulation.nproc}",
            rank_folder)
        shutil.copy(simson_conf_file,rank_folder)
    else:
        rank_folder = run_folder+f'/env_test_{conf.runner.rank:03d}'
        
        # make the env folder and copy all the necessary files
        if not os.path.exists(rank_folder):
            os.mkdir(rank_folder)

        if conf.runner.rewrite_input_files:
            simson_conf_file = write_simson_input_file(conf, 
                folder=rank_folder)        
        else:
            source_folder = run_folder+'/env_000'
            # source_folder = conf.runner.bin_root
            # make copies of the code from training environment
            print(f"copy binaries to env_test_{conf.runner.rank:03d}",
                flush=True)

            shutil.copy(source_folder+'/bla.i',rank_folder)
        
        shutil.copy(conf.runner.bin_root + \
            f"/bla_{conf.simulation.nx}x{conf.simulation.ny}x{conf.simulation.nz}_{conf.simulation.nproc}",
            rank_folder)
        
    
    # if not conf.runner.input_output_mapping:
        # Defintion of the environment
    env = SimsonEnv.parallel_env(conf=conf, rank_folder=rank_folder)

    # It is possible to evaluate a trained policy or a functional policy
    if conf.runner.learnt_policy==True:
        # Importing the required RL algorithm
        if conf.runner.RL_algorithm=='PPO':
            from stable_baselines3 import PPO as RL_algorithm
        elif conf.runner.RL_algorithm=='DDPG':
            from stable_baselines3 import DDPG as RL_algorithm

        # Definition of the agent
        if conf.runner.custom_policy:
            with open(conf.runner.policy_file) as file:
                policy_kwargs = yaml.load(file, Loader=yaml.FullLoader)
        else:
            policy_kwargs = {}

        # Load model from path
        # custom_objects is required because the action_space
        # is not correctly deserialized when loading from file 
        loaded_model = RL_algorithm.load(f"{run_folder}/"+\
            f"logs/{conf.runner.agent_run_name}-"+\
            f"{conf.runner.policy}",
            custom_objects={'action_space':env.action_space('jet_z0_x0')})

        # breakpoint()
        if not conf.runner.input_output_mapping:
            # Check if the action range is the same for the loaded model and the environment
            if loaded_model.action_space.low[0] != env.action_space('jet_z0_x0').low[0] or \
                    loaded_model.action_space.high[0] != env.action_space('jet_z0_x0').high[0]:
                env.rescale_actions = True
                print("WARNING: the environment action range is different from the agent "+\
                      "action range. The agents' actions are rescaled")
                rescale_factor = np.ones((2,),dtype=float)
                if loaded_model.action_space.low[0] != env.action_space('jet_z0_x0').low[0] and \
                        loaded_model.action_space.low[0] != 0:
                    rescale_factor[0] = env.action_space('jet_z0_x0').low[0]/loaded_model.action_space.low[0]
                if loaded_model.action_space.high[0] != env.action_space('jet_z0_x0').high[0] and \
                        loaded_model.action_space.high[0] != 0:
                    rescale_factor[1] = env.action_space('jet_z0_x0').high[0]/loaded_model.action_space.high[0]    
                env.rescale_factors.append(rescale_factor)
                print(rescale_factor)
                # Here we assume that both the max and min are different from zero
                if loaded_model.action_space.low[0] == 0 or env.action_space('jet_z0_x0').low[0] == 0 or \
                    loaded_model.action_space.high[0] == 0 or env.action_space('jet_z0_x0').high[0] == 0:
                    raise ValueError("The actions are assumed to be both positive or negative")

            # Vectorizing the environment
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")

            # Evaluate policy
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            states = None
            observations = env.reset()
            # Initialize the observation recorder if needed
            if conf.runner.vars_record:
                obs_rec = observations[np.newaxis,:]
                acts_rec = np.zeros((1,observations.shape[0],1))
                # XXX assumes 2 inputs for the agent 
                observations = observations[:,:2]

            for i in range(conf.runner.nb_interactions):
                actions, states = loaded_model.predict(observations, state=states, 
                        episode_start=episode_starts, deterministic=True)
                observations, rewards, dones, infos = env.step(actions)
                # breakpoint()
                if conf.runner.vars_record:
                    obs_rec = np.concatenate((obs_rec, observations[np.newaxis,:]),axis=0)
                    acts_rec = np.concatenate((acts_rec, actions[np.newaxis,:]),axis=0)
                    # XXX assumes 2 inputs for the agent 
                    observations = observations[:,:2]
            
            # Save the observation recorded
            if conf.runner.vars_record:
                np.savez(rank_folder+f'/vars_record_{int(time.time())}.npz',obs_rec=obs_rec,acts_rec=acts_rec)

        else:
            # Input/output mapping
            assert conf.simulation.nctrlz==conf.simulation.nz  \
               and conf.simulation.nctrlx==conf.simulation.nx, \
               "The input-output mapping is only possible if the number "+\
               "of agents is the same as the number of simulation points"
            
            # Auxiliary variables
            umin = -2*np.pi*conf.runner.ctrl_max_amp
            umax = 2*np.pi*conf.runner.ctrl_max_amp
            if conf.runner.normalize_input=="None":
                umin += 0.43220938584725477 # at y+=15!
                umax += 0.43220938584725477 # at y+=15!
            vmin = -conf.runner.ctrl_max_amp
            vmax = conf.runner.ctrl_max_amp

            # XXX input-output resolution hard-coded!
            # TODO scaling when ctrl_max_amp does not correspond with u_tau
            nptsx = 100//conf.simulation.nx*conf.simulation.nx #per plus unit
            nptsy = 100//conf.simulation.nz*conf.simulation.nz #per plus unit
            xp = int(np.ceil((umax-umin)/conf.runner.ctrl_max_amp)*nptsx)
            yp = int(np.ceil((vmax-vmin)/conf.runner.ctrl_max_amp)*nptsy)

            u = np.linspace(umin,umax,xp)
            v = np.linspace(vmin,vmax,yp)
            if conf.runner.normalize_input=="utau":
                # XXX only for u_tau scaling
                u /= conf.runner.ctrl_max_amp
                v /= conf.runner.ctrl_max_amp
            xr = np.arange(0,xp+1,conf.simulation.nx)
            zr = np.arange(0,yp+1,conf.simulation.nz)

            U,V = np.meshgrid(u,v)
            inputs_ = np.stack((V,U),axis=0)
            mapping = np.ndarray(U.shape)

            episode_starts = np.ones((conf.simulation.nx*conf.simulation.nz,), 
                                      dtype=bool)
            states = None
            observations = np.ndarray((conf.simulation.nx*conf.simulation.nz,
                                       conf.runner.npl_state,1,1))
            for i_x in range(len(xr)-1):
                for i_z in range(len(zr)-1):
                    observations_ = inputs_[:,zr[i_z]:zr[i_z+1],
                                              xr[i_x]:xr[i_x+1]]
                    # print(i_x,i_z)
                    for i_pl in range(conf.runner.npl_state):
                        observations[:,i_pl] = observations_[i_pl].reshape(-1,1,1)
                    actions, states = loaded_model.predict(observations, 
                        state=states, episode_start=episode_starts, 
                        deterministic=True)
                    
                    # breakpoint()
                    mapping[zr[i_z]:zr[i_z+1],xr[i_x]:xr[i_x+1]] = actions.reshape(conf.simulation.nz,
                                                                                   conf.simulation.nx)

            np.savez(f"{rank_folder}/input-output_{conf.runner.policy}.npz",mapping=mapping)
    else:
        # Load the policy function from script
        if conf.runner.policy.split('.')[-1]=='py':
            import importlib

            filename = conf.runner.policy.split('.')[0]
            policyname = filename.split('_')[1]
            loader = importlib.machinery.SourceFileLoader(filename, f'../data/{conf.runner.policy}')
            handle = loader.load_module(filename)
            policy = getattr(handle,policyname)
        else:
            raise ValueError("A functional policy needs to be provided if a learnt "+\
                            f"policy is not used (learnt_policy=={conf.learnt_policy})")

        # Function parameters can be loaded from a yml file, if needed
        from inspect import signature

        if len(signature(policy).parameters)>1:
            try: 
                with open(f"../data/kwargs-{policyname}.yml") as file:
                    policy_function_kwargs = yaml.load(file, Loader=yaml.FullLoader)
            except FileNotFoundError:
                policy_function_kwargs = {}
        
        if conf.runner.policy_function_kwargs == False:
            policy_function_kwargs = {}
        
        observations = env.reset()
        # Agents names, on which is possible to iterate (once supersuit wrappers are 
        # applied, the object "env" does not have the method ".possible_agents" anymore)
        possible_agents = env.possible_agents
        # Initialize the observation recorder if needed
        if conf.runner.vars_record:
            obs_rec = np.ndarray((len(possible_agents),)+observations["jet_z0_x0"].shape)
            obs_rec_ = np.ndarray((len(possible_agents),)+observations["jet_z0_x0"].shape)
            for i_a, agent in enumerate(possible_agents):
                obs_rec[i_a] = observations[agent]       
            obs_rec = obs_rec[np.newaxis,:]
            
            acts_rec_ = np.zeros((len(possible_agents),1))
            acts_rec = np.zeros((1,len(possible_agents),1))

        for i in range(conf.runner.nb_interactions):
            actions = {}
            for i_a,agent in enumerate(possible_agents):
                actions[agent] = policy(observations[agent],**policy_function_kwargs)
            observations,rewards,dones,infos = env.step(actions)
            if conf.runner.vars_record:
                for i_a, agent in enumerate(possible_agents):
                    obs_rec_[i_a] = observations[agent]
                    acts_rec_[i_a] = actions[agent]
                obs_rec = np.concatenate((obs_rec, obs_rec_[np.newaxis,:]),axis=0)
                acts_rec = np.concatenate((acts_rec, acts_rec_[np.newaxis,:]),axis=0)
            
        # Save the observation recorded
        if conf.runner.vars_record:
            np.savez(rank_folder+f'/vars_record_{int(time.time())}.npz',obs_rec=obs_rec, acts_rec=acts_rec)

    if not conf.runner.input_output_mapping:
        env.close()
