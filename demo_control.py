import time

from simulator import Vehicle, Park, Environment, initFrame
import numpy as np
import math

if __name__ == '__main__':
    vehL = 4
    steerVel = math.pi / 2
    speedAcc = 10
    steerT = 0.2

    init_x = 1.2
    init_y = 1.5
    init_theta = math.pi / 2

    environment = Environment(Vehicle(init_x, init_y, init_theta, 0, 0), Park())

    dt = 0.02
    t_max = 10
    n_frames = int(t_max / dt) + 1

    fig, ax = initFrame(figsize=(4, 4))
    no_collision = True
    speedIn = -1

    for t in np.linspace(0, t_max, n_frames):
        steerAngIn = math.pi / 4

        if environment.IsCollision:
            print('Collision !!! ')
            if no_collision:
                speedIn *= -1
                no_collision = False

        else:
            no_collision = True
        environment.vehicle.VehDynamics(steerAngIn, speedIn, dt, vehL, steerVel, speedAcc, steerT)
        environment.render(fig, ax)
