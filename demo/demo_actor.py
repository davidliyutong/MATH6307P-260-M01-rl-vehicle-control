from simulator import Vehicle, Environment, Park, initFrame, resetEnv
from network import ActorNetwork, CriticNetwork
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park())
    environment.seed(0)
    environment.reset()
    fig, ax = environment.InitFrame()
    actor_network = ActorNetwork(environment.observation_space, environment.action_space)
    actor_network.load_state_dict(torch.load('./checkpoint/actor_network.pth'))
    # actor_network = torch.load('./checkpoint/actor_network.pth')

    while True:
        # environment.render(fig, ax, mode='train')
        observation = environment.observation
        action = actor_network(observation)
        obs, rwd, is_done, info = environment.step(action)
        distances = environment.vehicle.GetAllRadarDistance(environment.park.segAll)
        print("distance: ", distances.squeeze())
        if environment.is_done:
            environment.reset()
            environment.render(fig, ax, mode='train')
