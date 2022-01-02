import time
from simulator import Vehicle, Park, Environment, initFrame, getFrame, resetEnvEval, resetEnvParked
import math
import pygame
import cv2


if __name__ == '__main__':
    LOG_OUTPUT = True
    screen = pygame.display.set_mode((400, 475), 0, 32)
    screen.fill([0, 0, 0])

    vehL = 4
    steerVel = math.pi / 2
    speedAcc = 10
    steerT = 0.2
    dt = 0.02

    environment = Environment(Vehicle(0, 0, 0, 0, 0), Park(), reset_fn=resetEnvParked)
    fig, ax = environment.InitFrame()

    no_collision = True
    steerAngIn = 0
    speedIn = 0
    restart = False

    while True:
        environment.reset_fn = resetEnvParked if environment.reset_fn == resetEnvEval else resetEnvEval
        time.sleep(0.1)
        environment.reset()

        steerAngIn = 0
        speedIn = 0
        no_collision = True

        if LOG_OUTPUT:
            log_filename = input("Input log filename:")
            if log_filename == '':
                log_filename = './log.txt'
            f = open(log_filename, 'w+')
        else:
            f = None

        while True:
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.transpose(frame)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))
            pygame.display.update()
            if f is not None:
                f.write(str(environment.vehicle.vehState.T) + '\n')

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if f is not None:
                        f.close()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        restart = True

            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_RETURN]:
                restart = True
                print("<ENTER>")
            if keys_pressed[pygame.K_ESCAPE]:
                if f is not None:
                    f.close()
                exit()
            if keys_pressed[pygame.K_w]:
                speedIn = min(1, speedIn + 0.01)
                print("<W>")
            if keys_pressed[pygame.K_a]:
                steerAngIn = min(environment.vehicle.maxSteerAng, steerAngIn + 0.01)
                print("<A>")
            if keys_pressed[pygame.K_s]:
                speedIn = max(-1, speedIn - 0.01)
                print("<S>")
            if keys_pressed[pygame.K_d]:
                steerAngIn = max(environment.vehicle.minSteerAng, steerAngIn - 0.01)
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

            if restart:
                restart = False
                break
        f.close()
