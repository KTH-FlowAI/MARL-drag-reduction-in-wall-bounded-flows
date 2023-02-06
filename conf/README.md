# Configuration settings

### DRL options
- **bin_root**: path to the solver binaries
- **ctrl_min_amp**: minimum control amplitude
- **ctrl_max_amp**: maximum control amplitude
- **ctrl_array_size**: action space size
- **npl_state**: number of inputs (*e.g.* the velocity components)

#### Reward-related parameters
- **rew_mode**: can be set to `Instantaneous` or `MovingAverage`
- **size_history**: only for reward with `MovingAverage`
- **partial_reward**: if the number of agent is the same as the number of simulation points in streamwise and spanwise direction, it is possible to choose between computing the reward in a single point or on the entire wall
- **nb_interactions**: number of actions per episode

#### Training-related parameters
- **agent_spec_file**: \[unused in the current implementation\]
- **nb_episodes**: number of learning episodes
- **ckpt_int**: interval between checkpoints
- **train_steps**: interval between network updates
- **gradient_steps**: number of mini-batch updates of the policy
- **RL_algorithm**: RL algorithm name, *e.g.* `DDPG`
- **custom_policy**: flag to enable the use of a custom network architecture
- **policy_file**: path to `yml` file that specifies the network architecture
- **action_noise**: when using DDPG (or similar algorithms), it determines the amplitude of the normal action noise applied during the policy learning
- **buffer_size**: when using DDPG (or similar algorithms), it defines the size of the replay buffer  
- **random_init**: defines whether we have a random initial condition or always the same. If set to `0`, the same condition is used, otherwise it has to be set equal to the number of available initial conditions. Files are expected to be name sequentially, *e.g.* `init_000.u`, `init_001.u`, etc.
    
#### Agent loading options
- **load_agent**: bool = False
- **agent_run_name**: during training is initialized as `0`, when evaluating, it is set to the timestamp of the training run to evaluate
- **normalize_input**: input scaling options
    + `utau` scaling with the control amplitude (zero-mean)
    + `std` scaling with the standard deviation of the input (zero-mean)   
    + `zero mean` only remove average value
- **npl_scale** : determines how many planes are scaled according the above option

#### Evaluation option
- **evaluation**: when set to `True`, the evaluation of the policy is performed
- **rewrite_input_files**: when set to `True`, rewrites the input files instead of copying them from the training folder
- **learnt_policy**: when set to `True`, the policy to be evaluated comes from a previous training, otherwise a functional policy is expected
- **input_output_mapping**: when set to `True`, the map between input and output is saved
- **vars_record**: when set to `True`, observations are saved in a npz file
- **rank**: number for the `env_test_***` folder
- **policy**: Either the name of the zip file or the `.py` file containing the policy as a function, named after the file (*e.g.* `policy-opposition.py` contains the function opposition). The policies are in the folder `data/`
- **policy_function_kwargs**: additional values to be specified for the functional policies

### Simulation

These parameters are written in the `bla.i` file that contains the runtime options for the solver. Here we describe the most important ones, most of the others are unused as they are necessary for the simulation of other flow cases.

- **init_field**: path to initial `.u` field file
- **tmax**: upper limit for the simulation time. This should be high enough to prevent the simulation to end before all the interactions have been performed
- **dt**: timestep for the solver, kept to a given number to maintain a regular actuation in time (using a timestep that is too high will give unphysical results, monitor CFL number during the simulation)
- **xl**: domain size
- **varsiz**: when set to `True`, initial fields with different resolution in the streamwise and spanwise directions can be used. The number of Chebyshev points in the wall-normal direction cannot be changed
- **ibc**: keep default to `4`, for the open channel flow boundary condition
- **wbci**: keep default to `7`

##### Blowing/suction parameters
- **nctrlz**: number of agents in the spanwise direction
- **nctrlx**: number of agents in the streamwise direction
- **npl**: number of sampled planes from the simulation. The first plane is sampled to compute the reward, the following ones represent the number of quantities sampled in the state (see also **simulation.npl_state** and **simulation.npl_state**) 
- **npl_list**: path `yml` that defines the sampled planes

##### MPI options
- **mpi_drlf**: enables MPI communication with the main python program (must be set to `True`)
- **ndrl**: number of iterations between the actions. Note that the solver provides a physical solution every 4 iterations, because of the Runge-Kutta scheme used for the time advancement, hence the value set here should be the number of intended iterations times 4 
- **npl_file**: when set to `True`, the sampled planes are saved as a separate files in the simulation folder. Files can be very large.
- **hostfile**: path to MPI hostfile

##### Additional information that is in par.f but not present in bla.i
- **Re_cl**: Reynolds number based on the centerline velocity
- **nproc**: number of cores on which the simulation is running
- **nx**: streamwise resolution
- **ny**: wall-normal resolution
- **nz**: spanwise resolution

##### Additional information regarding the size of the observable state
- **nzjet**: \[keep default to `1` in the current implementation\]
- **nxjet**: \[keep default to `1` in the current implementation\]
- **nxs**: \[recomputed at runtime in the current implementation, keep default to `nx`\]
- **nzs**: \[recomputed at runtime in the current implementation, keep default to `nz`\]

### Logging

- **run_name**: timestamp for the current learning run
- **group**: \[unused in the current implementation\]
- **notes**: \[unused in the current implementation\]
- **save_dir**: path to the folder where the runs are stored