# Data

This folder contains the initial fields as well as the functional policies. The initial fields provided with the repository are in the `baseline/` folder.

### Functional policies
These are deterministic policies that can be used instead of the DRL policies during evaluation, an example is the [opposition control policy](policy_opposition.py):

```python
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
```
The `.py` file is supposed to be called `policy_[policyname].py` and the included policy function is also named `[policyname]`. The first input is the observation of a single agent. The other arguments should have a default value.
The function returns the action of the agent.
The parameters of the function can be customized using an additional file `kwargs-[policyname].yml`. For opposition control it will be named `kwargs-opposition.yml`:
```python
limit_amp: True
```
in this example, the `limit_amp` value determines whether the action is clipped within a prescribed range.

The use of the functional policies is controlled in the configuration file by the following variables: **runner.learnt_policy**, **runner.policy** and **runner.policy_function_kwargs**