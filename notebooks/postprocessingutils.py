#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re

def eval_stats_reader(exp_dict,file,evaluation=False,ckpt=0):
    j = np.zeros((exp_dict['nb_init'],exp_dict['n_agents']), dtype=int)
    
    run_folder = '../runs/'
   
    # Change of retau and px
    re_hist = np.ndarray((exp_dict['nb_init'],2,5000000),dtype=float)
    stats_regex = re.compile("Calculate statistics")
    retau_regex = re.compile("(y=ny)")
    px_regex = re.compile("px =")
    

    for i_in in range(exp_dict['nb_init']):
        j_re = 0
        if not evaluation:
            dirr = run_folder+\
                   f"{exp_dict['timestamp']}/env_{i_in:03d}/"
        else:
            dirr = run_folder+\
                   f"{exp_dict['timestamp']}/env_test_{ckpt+i_in:03d}/"
        
        k = 0
        with open(dirr+file) as f:
            for line in f:
                k+=1
                # ==================================    
                # Check Re tau
                stats_check = stats_regex.search(line)
                if stats_check is not None:
                    re_hist[i_in,0,j_re] = float(line[28:56])

                retau_check = retau_regex.search(line)
                if retau_check is not None:
                    re_hist[i_in,1,j_re] = float(line[18:])
                
                px_check = px_regex.search(line)
                if px_check is not None:
                    # re_hist[i_in,2,j_re] = float(line[11:])
                    j_re += 1

        print(f'{k} lines checked for initial field {i_in},{j_re} values found')
    return re_hist[:,:,:j_re]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)