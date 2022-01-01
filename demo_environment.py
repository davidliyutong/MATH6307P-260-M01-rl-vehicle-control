from simulator import Vehicle, Environment, Park, initFrame, resetEnv
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park())
    environment.seed(0)
    environment.reset()
    fig, ax = environment.InitFrame()

    while True:
        # environment.render(fig, ax, mode='train')
        dummy_observation = environment.observation
        dummy_action = torch.randn(32)
        obs, rwd, is_done, info = environment.step(dummy_action)
        distances = environment.vehicle.GetAllRadarDistance(environment.park.segAll)
        print("distance: ", distances.squeeze())
        if environment.is_done:
            environment.reset()
            environment.render(fig, ax, mode='train')
