# PrioritizedExperienceReplay
Prioritized Experience Replay with support for vectorized environments, N-Step and memory-efficient framestacking.

This was tested with Python 3.11, with PyTorch 2.1.2+cu121 and numpy 1.26.1.

Example: 
```
import torch
import numpy as np
per = PER(size=2**20, device=torch.device("cuda:0"), n=3, envs=8, discount=0.99)

per.append(state, action, reward, next_state, done, trun, stream)
"""
All inputs here should be numpy arrays
:param state: observation of shape (framestack, imagex, imagey) (framestack=3 if using RGB)
:param action: Integer for which action was selected
:param reward: float for the reward
:param n_state: same as state, but the one after
:param done: boolean value for terminal
:param trun: boolean value for truncation
:param stream: which environment this was from (0 to num_envs - 1)

When using framestacking, the last frame in the stack (index -1),
should be the newest frame.
"""

states, actions, rewards, next_states, dones, weights = per.sample(batch_size)
"""
all returned values are torch tensors, moved to the device determined at init.
states: shape [batch, framestack, imagex, imagey], type torch float32 from 0-255 (when you do your learning
pass, remember to divide by 255)
actions: shape (batch). type int64
rewards: shape (batch). type float32
n_states: same as states. N-Step is applied, to these states will be N timesteps later. Remember to
apply your discount rate correctly!
dones: shape (batch). type boolean
weights: shape (batch). These are what your losses should be multipled by
"""

per.update_priorities(td_errors)
"""
:param priorities: new priorities that should be based on TD error from your learning update
Remember to move these back to CPU
"""
```
