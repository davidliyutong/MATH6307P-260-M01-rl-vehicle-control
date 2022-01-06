import time
from simulator import Vehicle, Park, Environment, getFrame, resetEnvEval, resetEnvParked
import math
import pygame
import cv2
import os
import datetime
import json

if __name__ == '__main__':
    """
    Keymap:
    W: Accelerate
    A: Steer-Left
    S: Deaccelerate
    D: Steer-Right
    SPACE: Break
    UP: 'FORCE' up
    LEFT: 'FORCE' left
    DOWN: 'FORCE' down
    RIGHT: 'FORCE' right
    Q: 'FORCE' left-rotate
    E: 'FORCE' right-rotate
    L: load a vehState(list)
    O: output current vehState to stdout
    """

    # >>> Configure log options
    LOG_OUTPUT = True
    LOG_DIR = './gamelog'
    # ---
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # <<<

    # >>> Configure display resolution
    resX = 400
    resY = 475
    # ---
    screen = pygame.display.set_mode((resX, resY), 0, 32)
    screen.fill([0, 0, 0])
    # <<<

    # >>> Simulation parameters, akin to MATLAB code
    vehL = 4
    steerVel = math.pi / 2
    speedAcc = 10
    steerT = 0.2
    dt = 0.02
    # <<<

    # >>> Initialize environment, canvas
    environment = Environment(Vehicle(0, 0, 0, 0, 0, minVel=-1, maxVel=1, maxSteerAng=1.5, minSteerAng=-1.5), Park(), reset_fn=resetEnvParked)
    fig, ax = environment.InitFrame()
    restart = False
    resetVehState = True
    # <<<

    while True:
        # >>> Reset Environment
        print(f"Resetting environment with resetVehState={resetVehState}")
        if resetVehState:
            # environment.reset_fn = resetEnvParked if environment.reset_fn == resetEnvEval else resetEnvEval
            environment.reset_fn = resetEnvEval
            time.sleep(0.1)
            environment.reset()
        else:
            resetVehState = True
        # <<<

        # >>> Reset Control
        steerAngIn = 0
        speedIn = 0
        no_collision = True
        n_frame = 0
        # <<<

        if LOG_OUTPUT:
            log_filename = os.path.join(LOG_DIR, str(datetime.datetime.utcnow().timestamp()) + ".json")
            log_tensor_buffer = []
            print(f"Writting to {log_filename}")
        else:
            log_filename = None
            log_tensor_buffer = None

        while True:
            n_frame += 1
            if log_tensor_buffer is not None:
                log_tensor_buffer.append({'state': environment.vehicle.vehState.T.numpy().tolist()[0], 'steerAngIn': float(steerAngIn), 'speedIn': float(speedIn), 'n_frame': n_frame})

            if environment.IsCollision:
                print('Collision !!! ')
                if no_collision:
                    environment.vehicle.vehState[4] *= -1
                    no_collision = False
                environment.vehicle.VehEvolution(dt, vehL)
                speedIn = environment.vehicle.vehState[4]
            else:
                no_collision = True
                environment.vehicle.VehDynamics(steerAngIn, speedIn, dt, vehL, steerVel, speedAcc, steerT)
            environment.Visualization(ax)
            frame = getFrame(fig)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.transpose(frame)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if log_filename is not None:
                        with open(log_filename, 'w+') as f:
                            json.dump(log_tensor_buffer, f, indent=4)
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        restart = True

            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_RETURN]:
                restart = True
                if log_filename is not None:
                    with open(log_filename, 'w+') as f:
                        json.dump(log_tensor_buffer, f, indent=4)
                print("<ENTER>")
            if keys_pressed[pygame.K_ESCAPE]:
                if log_filename is not None:
                    with open(log_filename, 'w+') as f:
                        json.dump(log_tensor_buffer, f, indent=4)
                exit()
            if keys_pressed[pygame.K_w]:
                speedIn = min(environment.vehicle.maxVel, speedIn + 0.02)
                print("<W>")
            if keys_pressed[pygame.K_a]:
                steerAngIn = min(environment.vehicle.maxSteerAng, steerAngIn + 0.02)
                print("<A>")
            if keys_pressed[pygame.K_s]:
                speedIn = max(environment.vehicle.minVel, speedIn - 0.02)
                print("<S>")
            if keys_pressed[pygame.K_d]:
                steerAngIn = max(environment.vehicle.minSteerAng, steerAngIn - 0.02)
                print("<D>")
            if keys_pressed[pygame.K_SPACE]:
                print("<SPACE>")
                speedIn = 0
            if keys_pressed[pygame.K_UP]:
                print("<UP>")
                environment.vehicle.vehState[1] += 0.005
            if keys_pressed[pygame.K_LEFT]:
                print("<LEFT>")
                environment.vehicle.vehState[0] -= 0.005
            if keys_pressed[pygame.K_DOWN]:
                print("<DOWN>")
                environment.vehicle.vehState[1] -= 0.005
            if keys_pressed[pygame.K_RIGHT]:
                print("<RIGHT>")
                environment.vehicle.vehState[0] += 0.005
            if keys_pressed[pygame.K_q]:
                print("<Q>")
                environment.vehicle.vehState[2] += 0.005
            if keys_pressed[pygame.K_e]:
                print("<E>")
                environment.vehicle.vehState[2] -= 0.005
            if keys_pressed[pygame.K_l]:
                print("<L>")
                resetVehState = False
                newStateStr = input("Input a list [init_x, init_y, init_theta, steerAng, speed]\n")
                try:
                    newState = eval(newStateStr)
                    assert isinstance(newState, list)
                    assert len(newState) == 5
                    assert all(map(lambda x: isinstance(x, float) or isinstance(x, int), newState))
                    for i in range(5):
                        environment.vehicle.vehState[i] = newState[i]
                except AssertionError as err:
                    print("Wrong input!!!", newStateStr)
                    print(err)
                    resetVehState = True
                except SyntaxError as err:
                    print("Wrong input!!!", newStateStr)
                    print(err)
                    resetVehState = True
                restart = True
            if keys_pressed[pygame.K_o]:
                print("vehState: " + "[" + ", ".join([str(float(environment.vehicle.vehState[i])) for i in range(5)]) + "]")
                time.sleep(0.1)

            if restart:
                restart = False
                break
