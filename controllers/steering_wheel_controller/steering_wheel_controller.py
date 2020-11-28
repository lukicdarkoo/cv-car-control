import os
import cv2
from vehicle import Driver
from visual_joystick import VisualJoystick


def main():
    # Initialize car
    driver = Driver()
    driver.setSteeringAngle(0.2)
    driver.setCruisingSpeed(10)
    timestep = int(driver.getBasicTimeStep())

    # Initialize camera
    camera = driver.getCamera('camera')
    camera.enable(timestep)

    # Initialize visual modules
    joystick = VisualJoystick()

    n_steps = 0
    while driver.step() != -1:
        n_steps += 1
        if n_steps % 20 == 0 and False:
            is_done, angle = joystick.get_control()
            if is_done:
                break

            if angle is not None:
                driver.setSteeringAngle(angle)
            else:
                driver.setSteeringAngle(0)


if __name__ == '__main__':
    main()
