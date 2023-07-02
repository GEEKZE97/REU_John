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


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)
