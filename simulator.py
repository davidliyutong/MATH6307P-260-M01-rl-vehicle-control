import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from matplotlib import patches
import math

class Vehicle:
    body_x = torch.tensor([2.5, 1.5, 1, -0.75, -0.75, -1.5, -1.5, -0.75, - 0.75, 1, 1.5, 2.5]).reshape(-1, 1)
    body_y = torch.tensor([0.8, 0.8, 1, 1, 0.8, 0.8, -0.8, -0.8, -1, -1, -0.8, -0.8]).reshape(-1, 1)
    front_x = torch.tensor([2.125, 1.875, 1.875, 2.5, 2.5, 2.25, 2.25, 2.5, 2.5, 1.875, 1.875, 2.125]).reshape(-1, 1)
    front_y = torch.tensor([0.75, 0.75, 1, 1, 0.75, 0.75, -0.75, -0.75, -1, -1, -0.75, -0.75]).reshape(-1, 1)
    rear_x = torch.tensor([-1.125, -0.875, -0.875, -1.5, -1.5, -1.25, -1.25, -1.5, -1.5, -0.875, -0.875, -1.125]).reshape(-1, 1)
    rear_y = torch.tensor([0.75, 0.75, 1, 1, 0.75, 0.75, -0.75, -0.75, -1, -1, -0.75, -0.75]).reshape(-1, 1)  # TODO This shape is releated with <TAG=0>

    def __init__(self, init_x, init_y, init_theta, steerAng, speed) -> None:
        self.vehState = torch.tensor([[init_x], [init_y], [init_theta], [steerAng], [speed]])

    def Draw(self, ax):
        """Draw the simulator
        """
        body_x = self.body_x
        body_y = self.body_y
        front_x = self.front_x
        front_y = self.front_y
        rear_x = self.rear_x
        rear_y = self.rear_y
        vehState = self.vehState

        vehOrg = min(self.body_x)
        previewL = previewW = 0

        body_x = body_x - vehOrg
        front_x = front_x - vehOrg
        rear_x = rear_x - vehOrg

        rotM1 = torch.tensor([[torch.cos(vehState[2]), -torch.sin(vehState[2])],
                              [torch.sin(vehState[2]), torch.cos(vehState[2])]])
        rotM2 = torch.tensor([[torch.cos(vehState[3]), -torch.sin(vehState[3])],
                              [torch.sin(vehState[3]), torch.cos(vehState[3])]])
        # TODO: Parallelization

        V_body1 = torch.cat([body_x, body_y], dim=1).T  # TODO This shape is releated with <TAG=0>
        V_al = torch.ones(V_body1.shape[1])
        V_body1 = rotM1 @ V_body1 + vehState[0:2] * V_al

        V_rear1 = torch.cat([rear_x, rear_y], dim=1).T  # TODO This shape is releated with <TAG=0>
        V_a1 = torch.ones(V_rear1.shape[1])
        V_rear1 = rotM1 @ V_rear1 + vehState[0:2] * V_a1

        V_b1 = torch.mean(front_x)
        V_front1 = torch.cat([front_x - V_b1, front_y], dim=1).T  # TODO This shape is releated with <TAG=0>
        V_a1 = torch.ones(V_front1.shape[1])
        V_front1 = rotM2 @ V_front1
        V_front1[0, :] += V_b1
        V_front1 = rotM1 @ V_front1 + vehState[0:2] * V_a1

        vehicle_patches = [
            patches.Polygon(V_rear1.numpy().T, True, color='black'),
            patches.Polygon(V_front1.numpy().T, True, color='black'),
            patches.Polygon(V_body1.numpy().T, True, color='red'),
        ]

        for patch in vehicle_patches:
            ax.add_patch(patch)

    def VehDynamics(self, steerAngIn, speedIn, dt, vehL, steerVel, speedAcc, steerT):
        expT = math.exp(-dt / steerT)  # TODO: Store this constant
        steerAngIdeal = torch.clip((1 - expT) * steerAngIn + expT * self.vehState[3], -math.pi / 4, math.pi / 4)
        vehState = self.vehState

        # vehicle steering angle evolution
        if (steerAngIdeal - vehState[3]) > steerVel * dt:
            vehState[3] += steerVel * dt
        elif (steerAngIdeal - vehState[3]) < -steerVel * dt:
            vehState[3] -= steerVel * dt
        else:
            vehState[3] = steerAngIdeal

        # vehicle velocity evolution
        if (speedIn - vehState[4]) > speedAcc * dt:
            vehState[4] += speedAcc * dt
        elif (speedIn - vehState[4]) < -speedAcc * dt:
            vehState[4] -= speedAcc * dt
        else:
            vehState[4] = speedIn

        vehState[0] = vehState[0] + vehState[4] * dt * torch.cos(vehState[2] + 0.5 * vehState[4] * dt * torch.tan(vehState[3]) / vehL)
        vehState[1] = vehState[1] + vehState[4] * dt * torch.sin(vehState[2] + 0.5 * vehState[4] * dt * torch.tan(vehState[3]) / vehL)
        vehState[2] = vehState[2] + vehState[4] * dt * torch.tan(vehState[3]) / vehL

        self.vehState = vehState


class Park(gym.Env):
    park_x = [-4, 0, 3]
    park_y = [-2.5, -1.5, 1.5, 7]

    def __init__(self) -> None:
        park_x = self.park_x
        park_y = self.park_y
        self.segAll = torch.tensor([[park_x[0], park_x[1], park_x[1], park_x[1], park_x[0], park_x[0], park_x[0], park_x[1], park_x[1], park_x[1], park_x[2], park_x[2]],
                                    [park_y[1], park_y[1], park_y[1], park_y[0], park_y[1], park_y[2], park_y[2], park_y[2], park_y[2], park_y[3], park_y[3], park_y[0]]])

        self.park_patches = [patches.Polygon(np.array([[park_x[0], park_x[1], park_x[1], park_x[0]], [park_y[0], park_y[0], park_y[1], park_y[1]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[0], park_x[1], park_x[1], park_x[0]], [park_y[2], park_y[2], park_y[3], park_y[3]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[2], park_x[2] + 0.5, park_x[2] + 0.5, park_x[2]], [park_y[0], park_y[0], park_y[3], park_y[3]]]).T, True, color='black'),
                             patches.Polygon(np.array([[park_x[0], park_x[0] - 0.5, park_x[0] - 0.5, park_x[0]], [park_y[0], park_y[0], park_y[3], park_y[3]]]).T, True, color='black')]

    def Draw(self, ax):
        """Draw the simulator
        """
        for patch in self.park_patches:
            ax.add_patch(patch)
        ax.set_xlim(self.park_x[0] - 0.5, self.park_x[2] + 0.5)
        ax.set_ylim(self.park_y[0], self.park_y[3])


class Environment(gym.Env):

    def __init__(self, vehicle: Vehicle, park: Park) -> None:
        self.vehicle = vehicle
        self.park = park

    def Visualization(self, ax):
        """Draw the simulator
        """
        self.park.Draw(ax)
        self.vehicle.Draw(ax)
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
        body_x = self.vehicle.body_x
        body_y = self.vehicle.body_y

        vehState: torch.Tensor = self.vehicle.vehState
        vehOrg: torch.Tensor = torch.min(body_x)
        vehCorners: torch.Tensor = torch.tensor([[max(body_x) - vehOrg, max(body_x) - vehOrg, 0, 0],
                                                 [max(body_y), min(body_y), min(body_y), max(body_y)]])
        rotM1: torch.Tensor = torch.tensor([[torch.cos(vehState[2]), -torch.sin(vehState[2])],
                                            [torch.sin(vehState[2]), torch.cos(vehState[2])]])
        vehCorners: torch.Tensor = rotM1 @ vehCorners + vehState[0:2].repeat(1, 4)

        for idx in range(0, self.park.segAll.shape[1], 2):
            if self.IsSegPolygonCrossed(self.park.segAll[:, idx:idx + 2], vehCorners):
                print('Collision !!! ')
                return True

        return False


if __name__ == '__main__':
    init_x = 1.2
    init_y = 1.5
    init_theta = math.pi / 2
    vehicle = Vehicle(init_x, init_y, init_theta, 0, 0)
    park = Park()
    environment = Environment(vehicle, park)
    environment.IsCollision
    environment.vehicle.VehDynamics(0, -1, 0.02, 4, 1.5708, 10, 0.2)
    environment.Visualization()
    print('end')
