# John Modl UROP Spring 2023: Towards Cooperative Intelligence in Multi-Agent Systems using Deep RL
# Adapted from the deepbots and deepworlds repositories: https://github.com/aidudezzz/deepbots
#                                                        https://github.com/aidudezzz/deepworlds
# Official citation:
#   @InProceedings{10.1007/978-3-030-49186-4_6,
#       author="Kirtas, M.
#       and Tsampazis, K.
#       and Passalis, N.
#       and Tefas, A.",
#       title="Deepbots: A Webots-Based Deep Reinforcement Learning Framework for Robotics",
#       booktitle="Artificial Intelligence Applications and Innovations",
#       year="2020",
#       publisher="Springer International Publishing",
#       address="Cham",
#       pages="64--75",
#       isbn="978-3-030-49186-4"
#   }
#----------------------------------------------------

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, reward, new_states, terminal
