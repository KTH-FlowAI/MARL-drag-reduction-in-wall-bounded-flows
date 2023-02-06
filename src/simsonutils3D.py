#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import yaml

#from configuration import Probe

def write_simson_input_file(conf,folder=''):
    if folder == '':
        simson_conf_file = conf.runner.bin_root+'/bla.i'
    else:
        simson_conf_file = folder+'/bla.i'
    print(f"writing input file {simson_conf_file}", flush=True)
    with open(simson_conf_file,'w') as f:
        f.write("# bla.i file written by the python wrapper\n")
        f.write("# Spatial channel flow simulation\n")
        f.write("20070716\n")
        f.write(f"{conf.simulation.init_field:<80}\n")
        f.write(f"{conf.simulation.end_field:<80}\n")
        
        f.write(f"{conf.simulation.tmax:.4f}\n")
        f.write(f"{conf.simulation.maxit}\n")
        f.write(f"{conf.simulation.cpumax}\n")
        f.write(f"{conf.simulation.wallmax}\n")
        
        f.write(f".{str(conf.simulation.write_inter).lower()}.\n")
        
        f.write(f"{conf.simulation.dt:.4f}\n")
        f.write(f"{conf.simulation.nst}\n")
        f.write(f"{conf.simulation.cflmaxin:.4f}\n")
        
        f.write(f"{conf.simulation.xl:.4f}\n")
        f.write(f".{str(conf.simulation.varsiz).lower()}.\n")
        f.write(f"{conf.simulation.rot:.4f}\n")
        f.write(f".{str(conf.simulation.cflux).lower()}.\n")
        if conf.simulation.cflux == False:
            f.write(f"{conf.simulation.retau:.4f}\n")
        
        f.write(f".{str(conf.simulation.pert).lower()}.\n")
        f.write(f"{conf.simulation.ibc}\n")
        f.write(f".{str(conf.simulation.cim).lower()}.\n")
        f.write(f".{str(conf.simulation.gall).lower()}.\n")
        f.write(f".{str(conf.simulation.suction).lower()}.\n")
        
        f.write(f".{str(conf.simulation.spat).lower()}.\n")
        if conf.simulation.spat == True:
            f.write(f".{str(conf.simulation.tabfre).lower()}.\n")
            f.write(f".{str(conf.simulation.rbfl).lower()}.\n")
            if conf.simulation.rbfl == True:
                f.write(f"{conf.simulation.namblf:<80}\n")
        
            # Fringe region parameters
            f.write(f"{conf.simulation.fmax:.4f}\n")
            f.write(f"{conf.simulation.fstart:.4f}\n")
            f.write(f"{conf.simulation.fend:.4f}\n")
            f.write(f"{conf.simulation.frise:.4f}\n")
            f.write(f"{conf.simulation.ffall:.4f}\n")
            
            f.write(f"{conf.simulation.ampob:.4f}\n")
            f.write(f"{conf.simulation.amp2d:.4f}\n")
            f.write(f".{str(conf.simulation.osmod).lower()}.\n")
            f.write(f".{str(conf.simulation.streak).lower()}.\n")
            f.write(f".{str(conf.simulation.waves).lower()}.\n")
        else:
            f.write(f"{conf.simulation.cdev:.4f}\n")
            
        f.write(f".{str(conf.simulation.sgs).lower()}.\n")
        f.write(f"{conf.simulation.isfd}\n")
        f.write(f"{conf.simulation.imhd}\n")
        
        # Localized perturbation parameters
        f.write("# Localized perturbation parameters\n")
        f.write(f"{conf.simulation.loctyp}\n")
        # f.write(f"{conf.simulation.ampx:.4f}\n")
        # f.write(f"{conf.simulation.ampy:.4f}\n")
        # f.write(f"{conf.simulation.ampz:.4f}\n")
        # f.write(f"{conf.simulation.xscale:.4f}\n")
        # f.write(f"{conf.simulation.xloc0:.4f}\n")
        # f.write(f"{conf.simulation.yscale:.4f}\n")
        # f.write(f"{conf.simulation.zscale:.4f}\n")
        # f.write(f"{conf.simulation.tscale:.4f}\n")
        # f.write(f"{conf.simulation.tomega:.4f}\n")
        # f.write(f"{conf.simulation.to:.4f}\n")
        
        f.write(f".{str(conf.simulation.tripf).lower()}.\n")

        # Blowing/suction parameters
        f.write("# Blowing/suction parameters\n")
        f.write(f"{conf.simulation.wbci}\n")
        if ((conf.simulation.wbci==6) or (conf.simulation.wbci==7)):
            f.write(f"{conf.simulation.nctrlz}\n")
            f.write(f"{conf.simulation.nctrlx}\n")
            f.write(f"{conf.simulation.oppamp:.4f}\n")
            f.write(f"{conf.simulation.localbs:.4f}\n")
            if conf.simulation.localbs > 0:
                f.write(f"{conf.simulation.xstart:.4f}\n")
                f.write(f"{conf.simulation.xend:.4f}\n")
                f.write(f"{conf.simulation.xrise:.4f}\n")
                f.write(f"{conf.simulation.xfall:.4f}\n")
            if conf.simulation.wbci==7:
                f.write(f"{conf.simulation.zchange:.4f}\n")
                f.write(f"{conf.simulation.xchange:.4f}\n")
            # f.write(f"{conf.simulation.x_rerise:.4f}\n")
        
        f.write(f"{conf.simulation.icfl}\n")
        f.write(f"{conf.simulation.iamp}\n")
        f.write(f".{str(conf.simulation.longli).lower()}.\n")
        f.write(f"{conf.simulation.iext}\n")
        f.write(f"{conf.simulation.ixys}\n")
        if conf.simulation.ixys > 0:
            f.write(f"{conf.simulation.namxys:<80}\n")
            f.write(f"{conf.simulation.ixyss}\n")
            f.write(f"{conf.simulation.txys:.4f}\n")
            f.write(f".{str(conf.simulation.corrf).lower()}.\n")
            f.write(f".{str(conf.simulation.corrf_x).lower()}.\n")
            f.write(f".{str(conf.simulation.serf).lower()}.\n")
            if conf.simulation.serf == True:
                f.write(f"{conf.simulation.namser:<80}\n")
                f.write(f"{conf.simulation.nser}\n")
        
        f.write(f"{conf.simulation.msave}\n")
        if conf.simulation.msave > 0:
            if conf.simulation.msave != 100000:
                for i_t in conf.simulation.mtsave:
                    f.write(f"{conf.simulation.i_t:.4f}\n")
                    f.write(f"t{conf.simulation.i_t}.u\n")
            else:
                f.write(f"{conf.simulation.ssave:.4f}\n")
                f.write(f"{conf.simulation.dsave}\n")
        f.write(f"{conf.simulation.mwave}\n")
        f.write(f"{conf.simulation.npl}\n")
        if conf.simulation.npl > 0:
            with open(conf.simulation.npl_list) as file:
                measure_list = yaml.load(file, Loader=yaml.FullLoader)
                f.write(f"{conf.simulation.ipl}\n")
            for i_pl in range(conf.simulation.npl):
                pl = measure_list['planes'][i_pl]
                f.write(f"{pl['pltype']}\n")
                f.write(f"{pl['var']}\n")
                f.write(f"{pl['coord']:.5f}\n")
                f.write(f"{pl['namfile']:<80}\n")
        
        f.write(f".{str(conf.simulation.mpi_drlf).lower()}.\n")
        if conf.simulation.mpi_drlf == True:
            f.write(f"{conf.simulation.ndrl}\n")
            f.write(f".{str(conf.simulation.npl_file).lower()}.\n")
    
    return simson_conf_file
