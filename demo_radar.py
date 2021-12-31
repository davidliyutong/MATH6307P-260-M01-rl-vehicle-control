from simulator import Vehicle, Environment, Park, initFrame
import matplotlib.pyplot as plt
if __name__ == '__main__':
    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park())
    environment.seed(0)
    environment.reset()

    fig, ax = environment.InitFrame()
    environment.render(fig, ax, mode='train')
    plt.show()


