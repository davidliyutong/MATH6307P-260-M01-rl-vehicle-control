import math

import torch
from matplotlib import patches


class Vehicle:
    body_x = torch.tensor([2.5, 1.5, 1, -0.75, -0.75, -1.5, -1.5, -0.75, - 0.75, 1, 1.5, 2.5]).reshape(-1, 1)
    body_y = torch.tensor([0.8, 0.8, 1, 1, 0.8, 0.8, -0.8, -0.8, -1, -1, -0.8, -0.8]).reshape(-1, 1)
    front_x = torch.tensor([2.125, 1.875, 1.875, 2.5, 2.5, 2.25, 2.25, 2.5, 2.5, 1.875, 1.875, 2.125]).reshape(-1, 1)
    front_y = torch.tensor([0.75, 0.75, 1, 1, 0.75, 0.75, -0.75, -0.75, -1, -1, -0.75, -0.75]).reshape(-1, 1)
    rear_x = torch.tensor([-1.125, -0.875, -0.875, -1.5, -1.5, -1.25, -1.25, -1.5, -1.5, -0.875, -0.875, -1.125]).reshape(-1, 1)
    rear_y = torch.tensor([0.75, 0.75, 1, 1, 0.75, 0.75, -0.75, -0.75, -1, -1, -0.75, -0.75]).reshape(-1, 1)  # TODO This shape is releated with <TAG=0>
    minSteerAng = -math.pi / 4
    maxSteerAng = math.pi / 4
    minVel = -1
    maxVel = 1

    def __init__(self, init_x, init_y, init_theta, steerAng, speed) -> None:
        self.vehState = torch.tensor([[float(init_x)], [float(init_y)], [float(init_theta)], [float(steerAng)], [float(speed)]], dtype=torch.float32)

        # Runtime constants
        self.vehOrg: float = float(min(self.body_x))
        self.vehCornersOriginal: torch.Tensor = torch.tensor([[max(self.body_x) - self.vehOrg, max(self.body_x) - self.vehOrg, 0, 0],
                                                              [max(self.body_y), min(self.body_y), min(self.body_y), max(self.body_y)]])

        self.v_body_x = self.body_x - self.vehOrg
        self.v_body_y = self.body_y
        self.v_front_x = self.front_x - self.vehOrg
        self.v_front_y = self.front_y
        self.v_rear_x = self.rear_x - self.vehOrg
        self.v_rear_y = self.rear_y

        self.V_a1 = {
            'V_body': torch.ones(self.v_body_x.shape[0]),
            'V_front': torch.ones(self.v_front_x.shape[0]),
            'V_rear': torch.ones(self.v_rear_x.shape[0])
        }
        self.V_b1 = {
            'V_front': float(torch.mean(self.v_front_x))
        }
        self.V_body = torch.cat([self.v_body_x, self.v_body_y], dim=1).T  # TODO This shape is releated with <TAG=0>
        self.V_front = torch.cat([self.v_front_x - self.V_b1['V_front'], self.v_front_y], dim=1).T  # TODO This shape is releated with <TAG=0>
        self.V_rear = torch.cat([self.v_rear_x, self.v_rear_y], dim=1).T  # TODO This shape is releated with <TAG=0>

    def Draw(self, ax):
        """Draw the simulator
        """
        vehState = self.vehState

        rotM1 = torch.tensor([[torch.cos(vehState[2]), -torch.sin(vehState[2])],
                              [torch.sin(vehState[2]), torch.cos(vehState[2])]])
        rotM2 = torch.tensor([[torch.cos(vehState[3]), -torch.sin(vehState[3])],
                              [torch.sin(vehState[3]), torch.cos(vehState[3])]])
        # TODO: Parallelization

        V_body1 = rotM1 @ self.V_body + vehState[0:2] * self.V_a1['V_body']
        V_rear1 = rotM1 @ self.V_rear + vehState[0:2] * self.V_a1['V_rear']

        V_front1 = rotM2 @ self.V_front
        V_front1[0, :] += self.V_b1['V_front']
        V_front1 = rotM1 @ V_front1 + vehState[0:2] * self.V_a1['V_front']



        vehicle_patches = [
            patches.Polygon(V_rear1.numpy().T, True, color='black'),
            patches.Polygon(V_front1.numpy().T, True, color='black'),
            patches.Polygon(V_body1.numpy().T, True, color='red'),
            patches.Polygon(self.vehCorners.numpy().T, color="blue", fill=False, linewidth=3),
            patches.Circle(self.pos, 0.2, color="blue", fill=True)
        ]

        for patch in vehicle_patches:
            ax.add_patch(patch)

    def VehDynamics(self, steerAngIn, speedIn, dt, vehL, steerVel, speedAcc, steerT):
        expT = math.exp(-dt / steerT)  # TODO: Store this constant
        steerAngIdeal = torch.clip((1 - expT) * steerAngIn + expT * self.vehState[3], self.minSteerAng, self.maxSteerAng)
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

    @property
    def vehCorners(self):
        vehState: torch.Tensor = self.vehState
        rotM1: torch.Tensor = torch.tensor([[torch.cos(vehState[2]), -torch.sin(vehState[2])],
                                            [torch.sin(vehState[2]), torch.cos(vehState[2])]])
        vehCorners: torch.Tensor = rotM1 @ self.vehCornersOriginal + vehState[0:2].repeat(1, 4)
        return vehCorners

    @property
    def velTensor(self):
        return self.vehState[4:5, :]

    @property
    def steerAngTensor(self):
        return self.vehState[3:4, :]

    @property
    def angTensor(self):
        return self.vehState[2:3, :]

    @property
    def posTensor(self):
        return self.vehState[0:2, :]

    @property
    def vel(self):
        return float(self.vehState[4, 0])

    @property
    def steerAng(self):
        return float(self.vehState[3, 0])

    @property
    def ang(self):
        return float(self.vehState[2, 0])

    @property
    def pos(self):
        return float(self.vehState[0, 0]), float(self.vehState[1, 0])
