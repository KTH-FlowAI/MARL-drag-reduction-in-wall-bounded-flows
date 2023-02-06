# Source code

This folder contains the code to train and evaluate the MARL agents:
- **configuration.py**: contains the default configuration of all the parameters
- **run.py** and **evaluate.py**: contain the routines to train and evaluate the agents, respectively. See the [README](https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows/blob/master/README.md) for instructions on their use
- **simson_marl.py**: contains the definition of the environment using the PettingZoo/Gym framework for DRL.
- **simsonutils3D.py**: contains additional routines to generate the solver input files, for instance.

A script for evaluation is also provided
- **evaluate-script.sh**: allows to run several evaluations at the same time. Three inputs are expected, the initial and final checkpoint and the highest number of initial condition to be tested, for example:
  ```bash
  bash ./evaluate-script.sh 46 46 0
  ```
  for a checkpoint 46 and a single initial condition.
  
The evaluation runs are saved in the `runs/[timestamp]/` folder.  