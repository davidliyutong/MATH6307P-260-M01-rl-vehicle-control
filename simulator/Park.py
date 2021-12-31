import gym
import numpy as np
import torch
from matplotlib import patches


class Park(gym.Env):
    park_x = [-4, 0, 3]
    park_y = [-2.5, -1.5, 1.5, 7]

    def __init__(self) -> None:
        park_x = self.park_x
        park_y = self.park_y
        self.segAll_ = torch.tensor([[park_x[0], park_x[1], park_x[1], park_x[1], park_x[0], park_x[0], park_x[0], park_x[1], park_x[1], park_x[1], park_x[2], park_x[2]],
                                     [park_y[1], park_y[1], park_y[1], park_y[0], park_y[1], park_y[2], park_y[2], park_y[2], park_y[2], park_y[3], park_y[3], park_y[0]]])
        self.segAll = torch.tensor([[[park_x[0], park_x[1]],
                                     [park_y[1], park_y[1]]],
                                    [[park_x[1], park_x[1]],
                                     [park_y[1], park_y[0]]],
                                    [[park_x[0], park_x[0]],
                                     [park_y[1], park_y[2]]],
                                    [[park_x[0], park_x[1]],
                                     [park_y[2], park_y[2]]],
                                    [[park_x[1], park_x[1]],
                                     [park_y[2], park_y[3]]],
                                    [[park_x[2], park_x[2]],
                                     [park_y[3], park_y[0]]],
                                    ])
        self.park_patches = [patches.Polygon(np.array([[park_x[0], park_x[1], park_x[1], park_x[0]], [park_y[0], park_y[0], park_y[1], park_y[1]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[0], park_x[1], park_x[1], park_x[0]], [park_y[2], park_y[2], park_y[3], park_y[3]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[2], park_x[2] + 0.5, park_x[2] + 0.5, park_x[2]], [park_y[0], park_y[0], park_y[3], park_y[3]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[0], park_x[0] - 0.5, park_x[0] - 0.5, park_x[0]], [park_y[0], park_y[0], park_y[3], park_y[3]]]).T, True, color='black')]

    def Draw(self, ax):
        """Draw the simulator
        """
        for patch in self.park_patches:
            ax.add_patch(patch)
