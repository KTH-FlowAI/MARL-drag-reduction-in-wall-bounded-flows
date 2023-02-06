#!/bin/bash -l
n_ckpti=$1
n_ckpt=$2
# number of initial conditions to check
nb_init=$3

tstamp="1672737751"
ckpt=6144000

ckpt_name=("rl_model_$(printf "%d" $(($ckpt*$n_ckpti)))_steps")

for kk in $(seq $(($n_ckpti+1)) $n_ckpt)
do 
   ckpt_name+=("rl_model_$(printf "%d" $(($ckpt*$kk)))_steps")
done

i_ckpt=1
for value in "${ckpt_name[@]}"
do
   for kk in $(seq 0 $nb_init)
   do
      python3 -m simson_MARL evaluate ../conf/MC16_trained.yml runner.policy="$value" runner.agent_run_name=$tstamp runner.rank="$(($ckpt*($i_ckpt-1+$n_ckpti)+$kk))" simulation.init_field=$(printf "../../../data/baseline/init_16x65x16_minchan_%03d.u" $kk) runner.rewrite_input_files=True runner.vars_record=True &
   done
   i_ckpt=$(($i_ckpt+1))
done

wait