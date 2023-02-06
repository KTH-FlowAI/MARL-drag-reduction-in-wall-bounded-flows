# Multi-agent deep reinforcement learning for turbulent drag reduction in channel flows

## Introduction
The code in this repository introduces a multi-agent reinforcement learning environment to design and benchmark control strategies aimed at reducing drag in turbulent open channel flow. The control is applied in the form of blowing and suction at the wall, while the observable state is configurable, allowing to choose different variables such as velocity and pressure, in different locations of the domain. The case is proposed as a benchmark for testing data-driven control strategies in three-dimensional turbulent wall-bounded flows. We provide a functional policy that implements opposition control, a state-of-the-art turbulence-control strategy from the literature, and a control policy obtained using deep deterministic policy gradient (DDPG). More details about the implementation and the results from the training of the agents are available in ["Deep reinforcement learning for turbulent drag reduction in channel flows", L. Guastoni, J. Rabault, P. Schlatter, H. Azizpour, R. Vinuesa](https://arxiv.org/abs/2301.09889) (2023)

## Pre-requisites
The code is supposed to be run within a [Singularity container `marl-channelflow`](https://kth-my.sharepoint.com/:u:/g/personal/guastoni_ug_kth_se/Ee72ob6qbLZHnC3-LEu2yeMBLyIKO4hChN3C-yc8SBqVpg?e=vQrihv). The code was tested successfully on SUSE Linux Enterprise Server 15 SP1, Debian 11 (bullseye) and Ubuntu 20.04.3 LTS, with Singularity>=3.10.
The solver is compiled to run on two cores, however at least four (physical) cores are recommended. See the **Known issues** section in case of MPI errors.

## Data
Initial conditions used for training (in the minimal channel) and testing (in both the minimal and larger channel) are included in the repository in order to be able to run the code. More initial conditions can be provided by getting in touch using the email address for correspondance in the paper.

## Training and inference
The training of the agents can be performed after downloading the Singularity container `marl-channelflow` and cloning this repository
```bash
git clone https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows.git
```
The following command should be launched from terminal:
```bash
singularity shell --bind /path/to/repository marl-channelflow
```
then, within singularity:
```bash
cd /path/to/repository/MARL-drag-reduction-in-wall-bounded-flows/src/
mpiexec -n 1 python3 -m simson_MARL run ../conf/learning_conf_filename.yml 
```

All the default parameters are defined in the [config file](https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows/blob/master/src/configuration.py). The `learning_conf_filename.yml` can be defined to customize the learning. See a detailed explanation of the parameters, as well as examples for the training in the minimal and larger channel in the `conf/` folder. 

Agent testing (*i.e.* a deterministic run) can be performed within the singularity container as follows:
```bash
cd /path/to/repository/MARL-drag-reduction-in-wall-bounded-flows/src/
mpiexec -n 1 python3 -m simson_MARL evaluate ../conf/testing_conf_filename.yml 
```

See the configuration files for evaluation in the minimal and larger channel in the `conf/` folder.

## Additional options

The default options can be changed using the configuration files as shown in the examples above, however the individual parameters can be overridden from command line as follows:

```bash
mpiexec -n 1 python3 -m simson_MARL run ../conf/learning_conf_filename.yml runner.nb_episodes=10 simulation.init_field='../../../data/baseline/new_init_field.u'
```

## Known issues
* Currently, we do not have the rights to distribute the source code of the solver, so the solver itself is provided only as a compiled executable. Please, get in touch using the email address for correspondance in the paper for additional information or compiled executables.
* The code provided is supposed to run on two cores, the code might have generate MPI errors in computers with four cores. The issue occurs when one instance of the environment is closed and a new one is opened. If the first operation is not concluded before the beginning of second one, there will not be enough MPI slots to host all the processes. Current workaround consists in creating a `hostfile` for MPI, that indicates a higher number of slots, for instance:
  ```
  localhost slots=8
  ```
  assuming that you are running the code locally on the computer.
  The training/evaluation can be then run including the hostfile option:
  ```bash
  mpiexec -n 1 --hostfile /full/path/to/hostfile python3 -m simson_MARL run ../conf/learning_conf_filename.yml 
  ```
  Make sure that the hostfile is in a position that is mounted within the Singularity container.
