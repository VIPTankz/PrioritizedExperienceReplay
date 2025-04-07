import numpy as np

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    def __init__(self, size, procgen=False):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
          self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, value):
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return self.sum_tree[indices], data_index, indices  # Return values, data indices, tree indices

    def total(self):
        return self.sum_tree[0]

class PER:
    def __init__(self, size, device, n, envs, gamma, alpha=0.6, beta=0.4, total_steps=50000000, framestack=4,
                 imagex=84, imagey=84, rgb=False, data_store_mult=1.35, whole_tree_weights=False):

        """
        :param size: Size of the replay buffer, must be a power of 2 (typically 2^20 is used)
        :param device: When sampling, the buffer will move the tensors onto your preferred device
        :param n: N-Step reinforcement learning. This buffer handles N-step for you
        :param envs: number of parallel environments
        :param gamma: your discount rate
        :param alpha: prioritization rate
        :param beta: Importance sampling bias annealing
        :param total_steps: Used to know how quickly to anneal beta towards 1
        :param framestack: Framestack of your environment (typically 4)
        :param imagex: (typically 84)
        :param imagey: (typically 84)
        :param rgb: False for greyscale, True for RGB images. Buffer does not support framestack+RGB.
        If RGB is used, framestack will not be used
        :param data_store_mult: Determines the size of your underlying data arrays. See below for an explanation
        :param whole_tree_weights: Determines if the weight normalization is based on the entire tree, or
        just the current batch. Setting True will slow performance, and rarely makes a significant different.
        """

        self.st = SumTree(size)
        self.data = [None for _ in range(size)]
        self.index = 0
        self.size = size

        # this is the number of frames, not the number of transitions
        # the technical size to ensure there are no errors with overwritten memory in theory is very high-
        # (2*framestack - overlap) * first_states + non_first_states
        # with N=3, framestack=4, size=1M, average ep length 20, we need a total frame storage of around 1.35M
        # this however is still pretty light given it uses discrete memory. Careful when using RGB though,
        # as you have to store every frame so memory usage will be notably higher.
        print("Thank you for using Prioritized Experience Replay")
        print("Please note, if you are using very short short episodes (average length < 20),"
              " you will need to increase data_store_mult")

        if rgb:
            self.storage_size = int(size * 4)
        else:
            self.storage_size = int(size * data_store_mult)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0

        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.imagex = imagex
        self.imagey = imagey

        self.total_steps = total_steps
        self.whole_tree_weights = whole_tree_weights

        self.max_prio = 1

        self.framestack = framestack

        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = device

        self.last_terminal = [True for i in range(envs)]
        self.tstep_counter = [0 for i in range(envs)]

        self.n_step = n
        self.state_buffer = [[] for i in range(envs)]
        self.reward_buffer = [[] for i in range(envs)]

        self.expected_state_shape = (self.framestack if not self.rgb else 3, self.imagex, self.imagey)

        if rgb:
            self.state_mem = np.zeros((self.storage_size, 3, self.imagex, self.imagey), dtype=np.uint8)
        else:
            self.state_mem = np.zeros((self.storage_size, self.imagex, self.imagey), dtype=np.uint8)
        self.action_mem = np.zeros(self.storage_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size, dtype=float)
        self.done_mem = np.zeros(self.storage_size, dtype=bool)
        self.trun_mem = np.zeros(self.storage_size, dtype=bool)

        # everything here is stored as ints as they are just pointers to the actual memory
        # reward contains N values. The first value contains the action. The set of N contains the pointers for both
        # the reward and dones
        self.trans_dtype = np.dtype([('state', int, self.framestack), ('n_state', int, self.framestack),
                                     ('reward', int, self.n_step)])

        self.blank_trans = (np.zeros(self.framestack, dtype=int), np.zeros(self.framestack, dtype=int),
                            np.zeros(self.n_step, dtype=int))

        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

        self.last_idxs = None

        # the "technically correct" way to do this is to use the min priority in the whole buffer. However,
        # we instead usually just use the min from each batch. In our experience, this makes effectively no difference
        # and is significantly faster.
        if self.whole_tree_weights:
            self.priority_min = [float('inf') for _ in range(2 * self.size)]

    def append(self, state, action, reward, n_state, done, trun, stream):
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

        # Check state and n_state are numpy arrays with the expected shape
        assert isinstance(state, np.ndarray), "state must be a numpy array."
        assert state.ndim == 3, f"state must have 3 dimensions, got {state.ndim}"
        assert state.shape == self.expected_state_shape, f"state must have shape {self.expected_state_shape}, but got {state.shape}"

        assert isinstance(n_state, np.ndarray), "n_state must be a numpy array."
        assert n_state.ndim == 3, f"n_state must have 3 dimensions, got {n_state.ndim}"
        assert n_state.shape == self.expected_state_shape, f"n_state must have shape {self.expected_state_shape}, but got {n_state.shape}"

        # Check that action is an integer.
        assert isinstance(action, int), "action must be an integer."

        # Check that reward is a float.
        assert isinstance(reward, (float, np.floating)), "reward must be a float."

        # Check that done and trun are booleans.
        assert isinstance(done, (bool, np.bool_)), "done must be a boolean."
        assert isinstance(trun, (bool, np.bool_)), "trun must be a boolean."

        # Check that stream is an integer.
        assert isinstance(stream, int), "stream must be an integer."

        # append to memory
        self.append_memory(state, action, reward, n_state, done, trun, stream)

        # append to pointer
        self.append_pointer(stream)

        if done or trun:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done

        # increase beta value
        self.beta = min(1., self.beta + (1 / self.total_steps))

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.size
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def append_pointer(self, stream):

        while len(self.state_buffer[stream]) >= self.framestack + self.n_step and len(self.reward_buffer[stream]) >= self.n_step:
            # First array in the experience
            state_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (starts after N frames)
            n_state_array = self.state_buffer[stream][self.n_step:self.n_step + self.framestack]

            # Reward array (first N rewards)
            reward_array = self.reward_buffer[stream][:self.n_step]

            # Add the experience to the list
            self.pointer_mem[self.point_mem_idx] = (np.array(state_array, dtype=int), np.array(n_state_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            if self.whole_tree_weights:
                self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))

            self.st.append(self.max_prio ** self.alpha)

            self.capacity = min(self.size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.size

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)
            self.beta = 0

    def finalize_experiences(self, stream):
        # Process remaining states and rewards at the end of an episode
        while len(self.state_buffer[stream]) >= self.framestack and len(self.reward_buffer[stream]) > 0:
            # First array in the experience
            first_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (Final `framestack` elements)
            second_array = self.state_buffer[stream][-self.framestack:]

            # Reward array
            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.extend([0])

            # Add the experience
            self.pointer_mem[self.point_mem_idx] = (np.array(first_array, dtype=int), np.array(second_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            if self.whole_tree_weights:
                self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))

            self.st.append(self.max_prio ** self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.size
            self.capacity = min(self.size, self.capacity + 1)

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(self, state, action, reward, n_state, done, trun, stream):

        if self.last_terminal[stream]:
            # add full transition
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # remember n_step is not applied in this memory
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0

        else:
            # just add relevant info
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    def sample(self, batch_size, count=0):
        """
        :param batch_size: The batch size you want to sample
        :return: tree_idxs, states, actions, rewards, n_states, dones, weights

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

        # get total sumtree priority
        p_total = self.st.total()

        # first use sumtree prios to get the indices
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length

        samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts

        prios, idxs, tree_idxs = self.st.find(samples)

        probs = prios / p_total

        # fetch the pointers by using indices
        pointers = self.pointer_mem[idxs]

        # Extract the pointers into separate arrays
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        # get state info
        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        # reward and dones just use the same pointer. actions just use the first one
        rewards = self.reward_mem[reward_pointers]
        dones = self.done_mem[reward_pointers]
        truns = self.trun_mem[reward_pointers]
        actions = self.action_mem[action_pointers]

        # apply n_step cumulation to rewards and dones
        if self.n_step > 1:
            rewards, dones = self.compute_discounted_rewards_batch(rewards, dones, truns)

        # Compute importance-sampling weights w
        weights = (self.capacity * probs) ** -self.beta

        if self.whole_tree_weights:
            prob_min = self.priority_min[1] / p_total
            max_weight = (prob_min * self.capacity) ** (-self.beta)
        else:
            max_weight = weights.max()

        weights = torch.tensor(weights / max_weight, dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch

        if torch.isnan(weights).any():
            # There is a very small chance to sample something outside of the currently filled range before the
            # buffer is full, causing a priority of 0 to be sampled, thus throwing a nan error.
            # In this case, we just sample again. If this happens more than a couple of times,
            # something else is probably broken
            if count >= 5:
                raise Exception("Weights Contained NaNs!")
            return self.sample(batch_size, count + 1)

        # move to pytorch GPU tensors
        states = states.to(torch.float32).to(self.device)
        n_states = n_states.to(torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        self.last_idxs = tree_idxs

        # return batch
        return states, actions, rewards, n_states, dones, weights

    # def compute_discounted_rewards_batch_no_loop(self, rewards_batch, dones_batch, truns_batch):
    #     """
    #     Compute discounted rewards for a batch using NumPy, in a vectorized manner.
    #     While its lovely that this doesn't use for loops, its way more complicated and not noticably faster
    #     than the for loop approach
    #
    #     Parameters:
    #       rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
    #       dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)
    #       truns_batch (np.ndarray): 2D array of truncation flags with shape (batch_size, n_step)
    #
    #     Returns:
    #       discounted_rewards (np.ndarray): 1D array of discounted rewards for each batch element,
    #                                        computed only up to (and including) the first step
    #                                        where either dones or truncation occurs.
    #       cumulative_dones (np.ndarray): 1D boolean array indicating for each sample whether
    #                                      the break was due to a done event.
    #     """
    #     batch_size, n_step = rewards_batch.shape
    #
    #     # Compute a boolean mask indicating where a break occurs
    #     break_mask = (dones_batch == 1) | (truns_batch == 1)  # shape: (batch_size, n_step)
    #
    #     # For each sample, find the first index where break_mask is True.
    #     # If no break occurs in a sample, set the break index to n_step - 1.
    #     break_exists = break_mask.any(axis=1)
    #     break_indices = np.empty(batch_size, dtype=int)
    #     if np.any(break_exists):
    #         # np.argmax returns the first index where the value is maximum.
    #         break_indices[break_exists] = np.argmax(break_mask[break_exists], axis=1)
    #     break_indices[~break_exists] = n_step - 1
    #
    #     # Create a mask that is True for all steps up to (and including) the break index.
    #     j_idx = np.arange(n_step)  # shape: (n_step,)
    #     valid_mask = j_idx[None, :] <= break_indices[:, None]  # shape: (batch_size, n_step)
    #
    #     # Create discount factors: [gamma^0, gamma^1, ..., gamma^(n_step-1)]
    #     discounts = self.gamma ** np.arange(n_step)  # shape: (n_step,)
    #
    #     # Compute the discounted rewards and zero out entries after the break index.
    #     discounted = rewards_batch * discounts[None, :]  # broadcast to (batch_size, n_step)
    #     discounted *= valid_mask.astype(rewards_batch.dtype)
    #     discounted_rewards = np.sum(discounted, axis=1)
    #
    #     # For cumulative_dones, take the 'dones' value at the break index for each sample.
    #     cumulative_dones = break_mask[np.arange(batch_size), break_indices]
    #
    #     return discounted_rewards, cumulative_dones

    def compute_discounted_rewards_batch(self, rewards_batch, dones_batch, truns_batch):
        """
        Compute discounted rewards for a batch of rewards and dones.

        Parameters:
        rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
        dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)

        Returns:
        np.ndarray: 1D array of discounted rewards for each element in the batch
        np.ndarray: 1D array of cumulative dones (True if any done is True in the sequence)
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(batch_size)
        cumulative_dones = np.zeros(batch_size, dtype=bool)

        for i in range(batch_size):
            cumulative_discount = 1
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                elif truns_batch[i, j] == 1:
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones

    def update_priorities(self, priorities):
        """
        :param priorities: new priorities that should be based on TD error from your learning update
        Remember to move these back to CPU
        """
        assert isinstance(priorities, np.ndarray), f"Priorities must be a numpy array, got {type(priorities)}"

        priorities = priorities + self.eps

        if self.whole_tree_weights:
            for idx, priority in zip(self.last_idxs, priorities):
                self._set_priority_min(idx - self.size + 1, sqrt(priority))

        if np.isnan(priorities).any():
            print("NaN found in priority!")
            print(f"priorities: {priorities}")

        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(self.last_idxs, priorities ** self.alpha)
