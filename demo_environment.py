from simulator import Vehicle, Environment, Park, initFrame, resetEnv
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park())
    environment.seed(0)
    environment.reset()
    fig, ax = initFrame(figsize=(4, 4))

    while True:
        environment.render(fig, ax)
        dummy_action = torch.randn(32)
        environment.step(dummy_action)
        if environment.is_done:
            environment.reset()

