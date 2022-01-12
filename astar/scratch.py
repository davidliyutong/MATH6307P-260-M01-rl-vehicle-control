from simulator import Environment, Vehicle, Park, resetEnvEval
from typing import Iterable, List
from copy import deepcopy
import torch
import math


def evolution(vehStates, dt, vehL):
    vehStates[:, 0] = vehStates[:, 0] + vehStates[:, 4] * dt * torch.cos(vehStates[:, 2] + 0.5 * vehStates[:, 4] * dt * torch.tan(vehStates[:, 3]) / vehL)
    vehStates[:, 1] = vehStates[:, 1] + vehStates[:, 4] * dt * torch.sin(vehStates[:, 2] + 0.5 * vehStates[:, 4] * dt * torch.tan(vehStates[:, 3]) / vehL)
    vehStates[:, 2] = vehStates[:, 2] + vehStates[:, 4] * dt * torch.tan(vehStates[:, 3]) / vehL
    return vehStates


def take_step(vehState: torch.Tensor, speed, steerAngs: torch.tensor, vehL=4, dt=0.02, simSteps=5):
    length = len(steerAngs)
    _vehState = vehState.repeat(1, length).T
    _vehState[:, 3] = steerAngs
    _vehState[:, 4] = speed

    for _ in range(simSteps):
        _vehState = evolution(_vehState, dt, vehL)
    return list(_vehState)


if __name__ == '__main__':
    # >>> Initialize environment, canvas
    environment = Environment(Vehicle(0, 0, 0, 0, 0, minVel=-1, maxVel=1, maxSteerAng=1.5, minSteerAng=-1.5), Park(), reset_fn=resetEnvEval)
    fig, ax = environment.InitFrame()
    # <<<
    GLOBAL_STEERANGLES = torch.linspace(-math.pi / 2, math.pi / 2, 5)
    GLOBAL_VEHL = 4
    GLOBAL_DT = 0.02
    GLOBAL_SIMSTEPS=5
    take_step(environment.vehicle.vehState, 1, GLOBAL_STEERANGLES, vehL=GLOBAL_VEHL, dt=GLOBAL_DT, simSteps=GLOBAL_SIMSTEPS)
