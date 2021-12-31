import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import math
import io
import cv2
import os
import random

from simulator.Park import Park
from simulator.Vehicle import Vehicle


def getFrame(fig) -> np.ndarray:
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def initFrame(figsize=(4, 4)):
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    return fig, ax


def resetEnv() -> tuple:
    """
    Return: init_x, init_y, inti_theta
    """
    X_CANDIDATES = np.linspace(0, 3, 601)
    Y_CANDIDATES = np.linspace(-1, 4, 601)
    THETA_CANDIDATES = np.linspace(math.radians(-30), math.radians(120), 64)

    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park())

    while True:
        init_x = np.random.choice(X_CANDIDATES)
        init_y = np.random.choice(Y_CANDIDATES)
        for init_theta in np.random.permutation(THETA_CANDIDATES):
            environment.vehicle.vehState = torch.tensor([[init_x], [init_y], [init_theta], [0], [0]], dtype=torch.float32)
            if environment.IsCollision:
                pass
            else:
                return init_x, init_y, init_theta


class Environment(gym.Env):

    def __init__(self, vehicle: Vehicle, park: Park, dt=0.02, vehL=4, steerVel=math.pi / 2, speedAcc=10, steerT=0.2, quantLevels=32) -> None:
        self.vehicle = vehicle
        self.park = park
        self.dt = dt
        self.vehL = vehL
        self.steerVel = steerVel
        self.speedAcc = speedAcc
        self.steerT = steerT

        self.quantLevels = quantLevels  # number of quantization levels, better be even number
        speedCandidates = torch.linspace(self.vehicle.minVel, self.vehicle.maxVel, quantLevels)
        steerAngCandidates = torch.linspace(self.vehicle.minSteerAng, self.vehicle.maxSteerAng, quantLevels)
        self.actionCandidates = torch.stack([speedCandidates, steerAngCandidates])

    def Visualization(self, ax):
        """Draw the simulator
        """
        ax.cla()
        self.park.Draw(ax)
        self.vehicle.Draw(ax)
        plt.axis("equal")

    @classmethod
    def IsTwoSegCrossed(cls, segA, segB):
        segV = segB[:, 0:1] - segA[:, 0:1]
        segM = torch.cat([segA[:, 1:2] - segA[:, 0:1], segB[:, 0:1] - segB[:, 1:2]], dim=1)
        if torch.abs(torch.linalg.det(segM)) < 1e-9:
            return False
        nmd = torch.linalg.inv(segM) @ segV
        if torch.all(nmd >= 0) and torch.all(nmd <= 1):
            return True
        else:
            return False

    @classmethod
    def IsSegPolygonCrossed(cls, seg, plg):
        plg_aug: torch.Tensor = torch.cat([plg, plg[:, 0:1]], dim=1)
        for idx in range(0, plg.shape[1]):
            if cls.IsTwoSegCrossed(seg, plg_aug[:, idx:idx + 2]):
                return True
        return False

    @property
    def IsCollision(self):
        vehCorners: torch.Tensor = self.vehicle.vehCorners

        for idx in range(0, self.park.segAll.shape[1], 2):
            if self.IsSegPolygonCrossed(self.park.segAll[:, idx:idx + 2], vehCorners):
                return True

        return False

    @property
    def reward(self):
        return 0

    @property
    def IsAccomplished(self):
        TARGET = torch.tensor([[-3.9], [0]])
        MARGIN = 2e-1
        if torch.sum(self.vehicle.posTensor - TARGET) < MARGIN:
            return True
        else:
            return False

    @property
    def IsOutOfBoundary(self):
        pos = self.vehicle.pos
        if pos[0] < -5 or pos[0] > 4 or pos[1] < -3 or pos[1] > 7:  # TODO: Magic Number
            return True
        else:
            return False

    @property
    def is_done(self):
        return self.IsCollision or self.IsAccomplished or self.IsOutOfBoundary

    @property
    def observation(self):
        # TODO: Finish this part
        return self.vehicle.vehState

    def render(self, fig, ax, *args, **kwargs):
        """
        fig: matplotlib fig
        ax: matplotlib axe
        """
        self.Visualization(ax)
        im = getFrame(fig)
        cv2.imshow('demo_control', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def step(self, action: torch.Tensor):
        """
        Action is 2D [steerAngIn, speedIn]
        """
        # Decompose action
        # [00 - 16] ==> steerAngIn
        # [16 - 32] ==> speedIn
        indices = action.reshape(2, -1).max(dim=1)[1]
        steerAngIn, speedIn = self.actionCandidates[0, indices[0]], self.actionCandidates[1, indices[1]]
        self.vehicle.VehDynamics(steerAngIn, speedIn, self.dt, self.vehL, self.steerVel, self.speedAcc, self.steerT)
        info = {'steerAngIn': steerAngIn, 'speedIn': speedIn}
        return self.observation, self.reward, self.is_done, info

    def reset(self):
        init_x, init_y, init_theta = resetEnv()
        self.vehicle.vehState = torch.tensor([[float(init_x)], [float(init_y)], [float(init_theta)], [0], [0]], dtype=torch.float32)
        return self.observation

    def seed(self, seed=None):
        seed = 0 if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        return None

    def close(self):
        return None

    @property
    def observation_space(self):
        return 32

    @property
    def action_space(self):
        return 2 * self.quantLevels
