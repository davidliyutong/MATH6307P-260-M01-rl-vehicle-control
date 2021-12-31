from simulator import Vehicle, Environment, Park, initFrame
from simulator import resetEnv
import matplotlib.pyplot as plt
import tqdm

if __name__ == '__main__':
    init_x, init_y, init_theta = resetEnv()
    environment = Environment(Vehicle(init_x, init_y, init_theta, 0, 0), Park())

    fig, ax = initFrame(figsize=(4, 4))
    environment.Visualization(ax)
    plt.show()

    with tqdm.tqdm(range(1)) as pbar:
        while True:
            resetEnv()
            pbar.update()
