from simulator import Vehicle, Environment, Park, initFrame, resetEnvEval
import matplotlib.pyplot as plt
if __name__ == '__main__':
    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park(), reset_fn=resetEnvEval)
    environment.seed(0)
    environment.reset()

    fig, ax = environment.InitFrame()
    environment.render(fig, ax, mode='train')
    print(environment.vehicle.GetAllRadarDistance(environment.park.segAll))
    plt.show()


