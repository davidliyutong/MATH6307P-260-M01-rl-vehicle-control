import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import math
import io
import cv2

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
    THETA_CANDIDATES = np.linspace(0, 2 * math.pi, 64)

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

    def __init__(self, vehicle: Vehicle, park: Park, dt=0.02, vehL=4, steerVel=math.pi / 2, speedAcc=10, steerT=0.2) -> None:
        self.vehicle = vehicle
        self.park = park
        self.dt = dt
        self.vehL = vehL
        self.steerVel = steerVel
        self.speedAcc = speedAcc
        self.steerT = steerT

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
    def is_done(self):
        return False

    @property
    def observation(self):
        # TODO: Finish this part
        return self.vehicle.vehState

    def render(self, fig, ax, *args, **kwargs):
        self.Visualization(ax)
        im = getFrame(fig)
        cv2.imshow('demo_control', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def step(self, action):
        """
        Action is 2D [steerAngIn, speedIn]
        """
        # Decompose action
        # [00 - 16] ==> steerAngIn
        # [16 - 32] ==> speedIn

        steerAngIn, speedIn = action
        vehicle.VehDynamics(steerAngIn, speedIn, self.dt, self.vehL, self.steerVel, self.speedAcc, self.steerT)
        observation = 0
        reward = 0
        done = False
        info = {}
        return observation, self.reward, self.is_done, info

    def reset(self):
        init_x, init_y, init_theta = resetEnv()
        self.vehicle.vehState = torch.tensor([[float(init_x)], [float(init_y)], [float(init_theta)], [0], [0]], dtype=torch.float32)
        return self.observation

    def seed(self, seed=None):
        return None

    def close(self):
        return None

    @property
    def observation_space(self):
        return

    @property
    def action_space_space(self):
        return


if __name__ == '__main__':
    init_x = 1.2
    init_y = 1.5
    init_theta = math.pi / 2
    vehicle = Vehicle(init_x, init_y, init_theta, 0, 0)
    park = Park()
    environment = Environment(vehicle, park)
    print(environment.IsCollision)
    environment.vehicle.VehDynamics(0, -1, 0.02, 4, 1.5708, 10, 0.2)
    environment.Visualization()
    print('end')
