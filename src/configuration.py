#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass, field
from typing import Optional, List

import time
import numpy as np
from omegaconf import OmegaConf

@dataclass
class Runner:
    bin_root: str = '../bin'
    #nb_ctrl_params: int = 1
    ctrl_min_amp: float = 0.0
    ctrl_max_amp: float = 1.0
    ctrl_array_size: int = 1

    npl_state: int = 2

    # Reward-related parameters
    rew_mode: str = 'Instantaneous'#'MovingAverage'# # Instantaneous or MovingAverage
    size_history: int = 10 # only for reward with MovingAverage
#    rew_filtering: bool = False
#    init_file_history: str = '../data/baseline/dudy_xz_yp0-t1800.npz'
    # ^only for reward with MovingAverage
    # if the number of agent is the same as the number of simulation points in 
    # streamwise and spanwise direction, it is possible to choose between computing
    # the reward in a single point or on the entire wall
    partial_reward: bool = True

#    rew_avg_time: int = 384 # number of iterations
    nb_interactions: int = 100
    
    # Training-related parameters
    agent_spec_file: str = '../conf/default_ppo.json'
    nb_episodes: int = 200
    ckpt_int: int = 8

    train_steps: int = 1
    gradient_steps: int = 0

    RL_algorithm: str = 'PPO'
    custom_policy: bool = False
    policy_file: str = '../conf/default_policy.yml'
    # only for DDPG
    action_noise: float = 0.01
    buffer_size: int = 5_000_000

    ## defines whether we have a random initial condition or always the same  
    random_init: int = 0
    
    # Agent loading options
    load_agent: bool = False
    agent_run_name: int = 0
    
    # Input scaling options
    # "utau"        scaling with the control amplitude (zero-mean)
    # "std"         scaling with the standard deviation of the input (zero-mean)   
    # "zero mean"   only remove average value
    normalize_input: str = "None"
    # determines how many planes are scaled according the above option
    npl_scale : int = 2

    # Evaluation option
    evaluation: bool = False
    # Rewrites probe.dat and bla.i instead of copying them from the 
    # env_000 folder 
    rewrite_input_files: bool = False
    learnt_policy: bool = False
    input_output_mapping: bool = False
    vars_record: bool = False
    # name for the env_test_*** folder
    rank: int = 0
    # Either the name of the zip file or the .py file containing the 
    # policy as a function, named after the file (e.g. policy-opposition.py 
    # contains the function opposition). The policies are in the folder
    # "data"
    policy: str = 'policy_opposition.py'
    policy_function_kwargs: bool = True

@dataclass
class Simulation:
    init_field: str = '../../../data/baseline/init.u'
    end_field: str = 'end.uu'
    
    tmax: float = 20000.0
    maxit: int = -40
    cpumax: int = -200
    wallmax: int = -200
    
    write_inter: bool = True
    
    dt: float = 0.007
    nst: int = 4
    cflmaxin: float = 0.8
    
    xl: float = 25.132741229
    varsiz: bool = False
    rot: float = 0
    cflux: bool = True
    retau: float = 180
    
    pert: bool = False
    # Open channel flow boundary condition
    ibc: int = 4
    cim: bool = False
    gall: bool = False
    suction: bool = False
    
    spat: bool = False
    cdev: float = 0
    sgs: bool = False
    isfd: int = 0
    imhd: int = 0
    
    # Localized perturbation parameters
    loctyp: int = 0
    # ampx: float = 0.0
    # ampy: float = 0.0
    # ampz: float = 0
    # xscale: float = 3
    # xloc0: float = 15
    # yscale: float = 0.1
    # zscale: float = 1
    # tscale: float = 100
    # tomega: float = 0
    # to: float = 1500
    
    tripf: bool = False
    wbci: int = 7
 
    # Blowing/suction parameters
    nctrlz: int = 1
    nctrlx: int = 1
    oppamp: float = 0
    localbs: float = 0
    xstart: float = 0.25
    xend: float = 3.39
    xrise: float = 0.5
    xfall: float = 0.5
    zchange: float = 4.0
    xchange: float = 4.0
    # x_restart: float = 70
    # x_rerise: float = 100
    
    icfl: int = 4
    iamp: int = 0
    longli: bool = False
    iext: int = 0
    ixys: int = 32
    namxys: str = 'xy.stat'
    ixyss: int = 64
    txys: float = 0#1500
    corrf: bool = False
    corrf_x: bool = False
    serf: bool = False
    
    # Timeseries output options
    namser: str = 'probes.dat'
    nser: int = 0
    
    msave: int = 0
    mtsave: Optional[List[int]] = None
    ssave: Optional[float] = 0
    dsave: Optional[int] = 10
    mwave: int = 0
    npl: int = 2
    ipl: int = 32 # TODO decide value
    npl_list: str = '../conf/default_planes.yml'
    
    # MPI options
    mpi_drlf: bool = True
    ndrl: int = 2048#192
    npl_file: bool = False
    hostfile: str = ''
    
    ## Additional information that is in par.f but not present in bla.i
    Re_cl: float = 2100
    nproc: int = 2
    nx: int = 64#128#32
    ny: int = 65#33
    nz: int = 64#128#32
    
    ## Additional information regarding the size of the observable state
    nzjet: int = 3
    nxjet: int = 3
    nxs: int = 64#128#32
    nzs: int = 64#128#32

@dataclass
class Logging:
    run_name: int = int(time.time())
    group: Optional[str] = None
    notes: Optional[str] = None
    save_dir: str = '../runs'

@dataclass
class Config:
    simulation: Simulation = Simulation()
    runner: Runner = Runner()
    logging: Logging = Logging()

def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("conf", help="Configuration")

    subparser.add_argument(
        "files_or_overrides",
        type=str,
        metavar="arg",
        nargs="*",
        help="Config YAML files e.g. `conf.yaml` or overrides e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=parse_cli)


def parse_cli(files_or_overrides: List[str], **ignored_kwargs):
    files = []
    overrides = []
    for x in files_or_overrides:
        if "=" in x:
            overrides.append(x)
        elif x.endswith((".yaml", ".yml")):
            files.append(x)
        else:
            raise ValueError(f"Unrecognized: {x}")
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        *map(OmegaConf.load, files),
        OmegaConf.from_dotlist(overrides),
    )
    print(OmegaConf.to_yaml(conf, resolve=True))


if __name__ == "__main__":
    print(parse_cli())

