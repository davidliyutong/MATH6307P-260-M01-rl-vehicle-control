from simulator import Vehicle, Environment, Park, resetEnvEval
import sys
import math
import json
import tqdm

if __name__ == '__main__':
    LOG_FILENAME = sys.argv[1]
    with open(LOG_FILENAME) as f:
        gamelog = json.load(f)
    assert isinstance(gamelog, list)

    # >>> Simulation parameters, akin to MATLAB code
    vehL = 4
    steerVel = math.pi / 2
    speedAcc = 10
    steerT = 0.2
    dt = 0.02
    # <<<

    # >>> Initialize environment, canvas
    environment = Environment(Vehicle(0, 0, 0, 0, 0, minVel=-1, maxVel=1, maxSteerAng=1.5, minSteerAng=-1.5), Park(), reset_fn=resetEnvEval)
    environment.reset()
    fig, ax = environment.InitFrame()
    restart = False
    resetVehState = True
    # <<<

    no_collision = True
    with tqdm.tqdm(range(len(gamelog))) as pbar:
        for entry in gamelog:
            steerAngIn = entry['steerAngIn']
            speedIn = entry['speedIn']

            if environment.IsCollision:
                if no_collision:
                    environment.vehicle.vehState[4] *= -1
                    no_collision = False
                environment.vehicle.VehEvolution(dt)
            else:
                no_collision = True
                environment.vehicle.VehDynamics(steerAngIn, speedIn, dt)

            if pbar.n % 10 == 0:
                environment.render(fig, ax)
            pbar.set_description(f"steerAngIn={steerAngIn}, speedIn=e{speedIn}")
            pbar.update()
