import time

from simulator import  Vehicle, Park, Environment
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import cv2

if __name__ == '__main__':
    vehL = 4
    steerVel = math.pi / 2
    speedAcc = 10
    steerT = 0.2

    init_x = 1.2
    init_y = 1.5
    init_theta = math.pi / 2

    vehicle = Vehicle(init_x, init_y, init_theta, 0, 0)
    park = Park()
    environment = Environment(vehicle, park)

    no_collision = True
    speedIn = -1
    dt = 0.02
    t_max = 10
    n_frames = int(t_max / dt) + 1

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)

    for t in np.linspace(0, t_max, n_frames):
        steerAngIn = math.pi / 4

        if environment.IsCollision:
            if no_collision:
                speedIn *= -1
                no_collision = False

        else:
            no_collision = True
        environment.vehicle.VehDynamics(steerAngIn, speedIn, dt, vehL, steerVel, speedAcc, steerT)

        ax.cla()
        environment.Visualization(ax)
        plt.axis("equal")
        # plt.draw()
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        cv2.imshow('demo', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break