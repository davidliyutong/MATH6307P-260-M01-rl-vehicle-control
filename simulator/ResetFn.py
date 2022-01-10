import math

import numpy as np
import torch

from . import Environment, Vehicle, Park


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