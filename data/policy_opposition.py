#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

def opposition(observation,alpha=1,limit_amp=False,amp=0.04285714285714286):
    """
    Implementation of the opposition control in the
    MARL environment framework
    """
    # Determine the size of the observation field
    npl = observation.shape[0]
    nzs = observation.shape[1]
    nxs = observation.shape[2]
    
    # Opposition control is based on the wall-normal velocity,
    # it is assumed that is the first field in the observations
    action = -alpha*np.mean(observation[0])
    
    # Limiter of the action intensity
    if limit_amp:
        action = np.maximum(action,-amp)
        action = np.minimum(action,amp)

    return action