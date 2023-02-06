#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List

import torch
# print(torch.get_num_threads())
torch.set_num_threads(8)

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker

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

from configuration import Config
from simsonutils3D import write_simson_input_file
# from gym_simson.envs import SimsonEnv
# sys.path.insert(0, '../envs')
# breakpoint()
import simson_marl as SimsonEnv
# breakpoint()

os.environ["CUDA_VISIBLE_DEVICES"]=""

def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("run", help="Run SIMSON")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=run)

def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf

def run(conf_file, overrides, **ignored_kwargs):
    
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
    
    # if conf.runner.parallel == False:
    if conf.runner.rewrite_input_files or not conf.runner.evaluation:
        
        # Write bla.i file in the bin_root folder
        simson_conf_file = write_simson_input_file(conf)
    
    if not conf.runner.evaluation:
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
        if not conf.runner.rewrite_input_files:
            source_folder = run_folder+'/env_000'
        else:
            source_folder = conf.runner.bin_root
            
        # make copies of the code from training environment
        print(f"copy binaries to env_test_{conf.runner.rank:03d}",
                flush=True)
        rank_folder = run_folder+f'/env_test_{conf.runner.rank:03d}'
        
        # make the env folder and copy all the necessary files
        if not os.path.exists(rank_folder):
            os.mkdir(rank_folder)
        
        shutil.copy(conf.runner.bin_root + \
            f"/bla_{conf.simulation.nx}x{conf.simulation.ny}x{conf.simulation.nz}_{conf.simulation.nproc}",
            rank_folder)
        shutil.copy(source_folder+'/bla.i',rank_folder)
        
    # Defintion of the environment
    env = SimsonEnv.parallel_env(conf=conf, rank_folder=rank_folder)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")

    # Definition of the agent
    if conf.runner.custom_policy:
        with open(conf.runner.policy_file) as file:
            policy_kwargs = yaml.load(file, Loader=yaml.FullLoader)
    else:
        policy_kwargs = {}

    if conf.runner.RL_algorithm=='PPO':
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', env, verbose=3,policy_kwargs=policy_kwargs,
                                      n_steps=conf.runner.train_steps)
    elif conf.runner.RL_algorithm=='DDPG' or conf.runner.RL_algorithm=='TD3':
        if conf.runner.RL_algorithm=='DDPG':
            from stable_baselines3 import DDPG as RLA
        else:
            from stable_baselines3 import TD3 as RLA
        from stable_baselines3.common.noise import NormalActionNoise
        
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
                                         sigma=conf.runner.action_noise*np.ones(n_actions))
        
        if conf.runner.gradient_steps == 0:
            conf.runner.gradient_steps = conf.simulation.nctrlz*\
                                         conf.simulation.nctrlx*\
                                         conf.runner.train_steps   

        model = RLA('MlpPolicy', env, verbose=3,
                                       buffer_size=conf.runner.buffer_size,
                                       policy_kwargs=policy_kwargs,
                                       action_noise=action_noise,
                                       train_freq=(conf.runner.train_steps, "step"),
                                       gradient_steps=conf.runner.gradient_steps)
                                       # the update unit (train_freq) needs to be "step"
                                       # to work with vectorized environments  
    # Definition of the learning callbacks
    checkpoint_callback = CheckpointCallback(
                                save_freq=conf.runner.nb_interactions*conf.runner.ckpt_int, 
                                save_path=run_folder+'/logs/',
                                name_prefix=f'{conf.logging.run_name}-rl_model')

    # Actual training
    model.learn(total_timesteps=conf.simulation.nctrlx*conf.simulation.nctrlz*\
                                conf.runner.nb_interactions*conf.runner.nb_episodes,
                callback=checkpoint_callback)
    model.save(f"policy_{conf.runner.agent_run_name}")
    
