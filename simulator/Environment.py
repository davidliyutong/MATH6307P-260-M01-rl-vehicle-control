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
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    return im


def initFrame(figsize=(4, 4)):
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    return fig, ax


def resetEnv() -> tuple:
    """
    Return: init_x, init_y, inti_theta, init_steer_ang, init_speed
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
                return init_x, init_y, init_theta, 0, 0

def resetEnvVel() -> tuple:
    """
    Return: init_x, init_y, inti_theta, init_steer_ang, init_speed
    """
    X_CANDIDATES = np.linspace(0.5, 2.5, 301)
    Y_CANDIDATES = np.linspace(1, 4, 301)
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
                return init_x, init_y, init_theta, 0, float(np.random.randn(1))


def resetEnvEval() -> tuple:
    """
    Return: init_x, init_y, inti_theta, init_steer_ang, init_speed
    """
    init_x = 1.2
    init_y = 1.5
    init_theta = math.pi / 2

    return init_x, init_y, init_theta, 0, 0

def resetEnvParked() -> tuple:
    """
    Return: init_x, init_y, inti_theta, init_steer_ang, init_speed
    """
    init_x = -3.9
    init_y = -0.4
    init_theta = 0

    return init_x, init_y, init_theta, 0, 0

class Environment(gym.Env):
    TARGET_POS = torch.tensor([[-3.9], [0]])
    TARGET_ANG = torch.tensor([[0]])
    MARGIN = 1

    def __init__(self,
                 vehicle: Vehicle,
                 park: Park,
                 dt=0.02,
                 vehL=4,
                 steerVel=math.pi / 2,
                 speedAcc=10,
                 steerT=0.2,
                 quantLevels=8,
                 reset_fn=resetEnv,
                 maxSteps=1000,
                 device=torch.device('cpu')) -> None:
        self.vehicle = vehicle
        self.park = park
        self.dt = dt
        self.vehL = vehL
        self.steerVel = steerVel
        self.speedAcc = speedAcc
        self.steerT = steerT
        self.reset_fn = reset_fn
        self.maxSteps = maxSteps
        self.n_steps = 0
        self.device = device
        self.no_collision = True

        self.quantLevels = quantLevels  # number of quantization levels, better be even number
        speedCandidates = torch.linspace(self.vehicle.minVel, self.vehicle.maxVel, quantLevels)
        steerAngCandidates = torch.linspace(self.vehicle.minSteerAng, self.vehicle.maxSteerAng, quantLevels)
        self.actionCandidates = torch.stack([speedCandidates, steerAngCandidates])

    @classmethod
    def InitFrame(cls):
        fig = plt.figure(figsize=(4, 4.75), dpi=100)
        ax = fig.add_subplot(111)
        return fig, ax

    @classmethod
    def GetFrame(cls, fig) -> np.ndarray:
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im

    def Visualization(self, ax, mode='eval'):
        """Draw the simulator
        """
        ax.cla()
        self.park.Draw(ax)
        if mode == 'eval':
            self.vehicle.Draw(ax)
        elif mode == 'train':
            self.vehicle.DrawTrain(ax)
        ax.set_xlim(self.park.park_x[0] - 0.5, self.park.park_x[2] + 0.5)
        ax.set_ylim(self.park.park_y[0], self.park.park_y[3])

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

        for idx in range(0, self.park.segAll.shape[0]):
            if self.IsSegPolygonCrossed(self.park.segAll[idx, :, :], vehCorners):
                return True

        return False

    @property
    def reward(self):
        radar_readings = self.vehicle.GetAllRadarDistance(self.park.segAll)
        pos_err: torch.Tensor = self.TARGET_POS - self.vehicle.vehState[0:2, :]
        ang_err = self.TARGET_ANG - self.vehicle.vehState[2:3, :]
        steerAng = self.vehicle.vehState[3:4]
        vehVel = self.vehicle.vehState[4:5]
        is_parked = 1 if self.IsParked else 0
        is_collision = 1 if self.IsCollision else 0
        rwd = 0 \
              + 2 * torch.exp(-(0.05 * torch.sum(pos_err ** 2))) \
              + 0.5 * torch.exp(-40 * ang_err ** 2) \
              + 0.05 * steerAng ** 2 \
              + 0.05 * vehVel ** 2 \
              + 100 * is_parked \
              + -50 * is_collision \
              + 0.1 * torch.log10(radar_readings.sum())
        return rwd

    @property
    def IsParked(self):
        if torch.sum(self.vehicle.posTensor - self.TARGET_POS) < self.MARGIN:
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
        return (self.n_steps > self.maxSteps) or self.IsParked or self.IsOutOfBoundary

    @property
    def observation(self):
        vehicle_state = self.vehicle.vehState
        pos_err: torch.Tensor = self.TARGET_POS - self.vehicle.vehState[0:2, :]
        ang = torch.tensor([[math.sin(self.vehicle.vehState[2:3, :])], [math.cos(self.vehicle.vehState[2:3, :])]])
        radar_readings = self.vehicle.GetAllRadarDistance(self.park.segAll)
        obs = torch.cat([vehicle_state, pos_err, ang, radar_readings]).T
        return obs

    def render(self, fig, ax, mode='eval', *args, **kwargs):
        """
        fig: matplotlib fig
        ax: matplotlib axe
        """
        self.Visualization(ax, mode)
        im = getFrame(fig)
        cv2.imshow('demo_control', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return im

    def step(self, action: torch.Tensor):
        """
        Action is 2D [steerAngIn, speedIn]
        """
        # Decompose action
        # [00 - 16] ==> steerAngIn
        # [16 - 32] ==> speedIn
        self.n_steps += 1
        indices = action.reshape(2, -1).max(dim=1)[1]
        steerAngIn, speedIn = self.actionCandidates[0, indices[0]], self.actionCandidates[1, indices[1]]
        info = {'steerAngIn': steerAngIn, 'speedIn': speedIn}

        # IsDone
        is_collision = self.IsCollision
        # if is_collision:
        #     if self.no_collision:
        #         self.vehicle.vehState[4] *= -1
        #         self.no_collision = False
        #     self.vehicle.VehEvolution(self.dt, self.vehL)
        # else:
        #     self.no_collision = True
        #     self.vehicle.VehDynamics(steerAngIn, speedIn, self.dt, self.vehL, self.steerVel, self.speedAcc, self.steerT)
        self.vehicle.VehDynamics(steerAngIn, speedIn, self.dt, self.vehL, self.steerVel, self.speedAcc, self.steerT)

        is_parked = self.IsParked
        is_out_of_boundary = self.IsOutOfBoundary
        is_done = (self.n_steps >= self.maxSteps) or is_parked or is_out_of_boundary

        # Observation
        # target_pos = self.TARGET_POS
        vehicle_state = self.vehicle.vehState
        pos_err: torch.Tensor = self.TARGET_POS - self.vehicle.vehState[0:2, :]
        ang = torch.tensor([[math.sin(self.vehicle.vehState[2:3, :])], [math.cos(self.vehicle.vehState[2:3, :])]])
        radar_readings = self.vehicle.GetAllRadarDistance(self.park.segAll)
        obs = torch.cat([vehicle_state, pos_err, ang, radar_readings]).T

        # Reward
        ang_err = self.TARGET_ANG - self.vehicle.vehState[2:3, :]
        steerAng = self.vehicle.vehState[3:4]
        vehVel = self.vehicle.vehState[4:5]
        rwd = 2 * torch.exp(-(0.05 * torch.sum(pos_err ** 2))) \
              + 0.5 * torch.exp(-400 * ang_err ** 2) \
              + 0.05 * steerAng ** 2 \
              + 0.05 * vehVel ** 2 \
              + is_parked * 100 \
              - is_collision * 50 \
              + 0.1 * torch.log10(radar_readings.sum())

        # return self.observation, self.reward, self.is_done, info
        return obs.to(self.device), rwd.to(self.device), bool(is_done), info

    def reset(self):
        self.no_collision = True
        self.n_steps = 0
        init_x, init_y, init_theta, init_steer_ang, init_speed = self.reset_fn()
        self.vehicle.vehState = torch.tensor([[float(init_x)], [float(init_y)], [float(init_theta)], [0], [0]], dtype=torch.float32)
        return self.observation.to(self.device)

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
        return 5 + 2 + 2  + self.vehicle.radarNum

    @property
    def action_space(self):
        return 2 * self.quantLevels
