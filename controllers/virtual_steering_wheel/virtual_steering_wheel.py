import os
import cv2
from vehicle import Driver
from wheel_detector import get_objects_from_net, filter_overlaps, \
    filter_by_confidence, draw_rectangles, get_wheel_properties, draw_wheel


driver = Driver()
driver.setSteeringAngle(0.2)
driver.setCruisingSpeed(10)

project_dir = os.path.dirname(__file__)
capture = cv2.VideoCapture(0)
wheel_image = cv2.imread(os.path.join(project_dir, 'steering_wheel.png'), -1)

# Load YOLO
net = cv2.dnn.readNet(
    '/home/lukic/Downloads/yolov3-tiny_train (5).backup', os.path.join(project_dir, 'yolov3-tiny.cfg'))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1]
                 for i in net.getUnconnectedOutLayers()]


n_steps = 0
while driver.step() != -1:
    n_steps += 1
    if n_steps % 10 == 0:    
        _, image = capture.read()
        image = cv2.flip(image, 1)
        objects = get_objects_from_net(net, output_layers, image)
        objects = filter_overlaps(objects)
        objects = filter_by_confidence(objects)

        draw_rectangles(objects, image)

        if len(objects) == 2:
            angle, _, _, diameter = get_wheel_properties(objects)
            draw_wheel(image, wheel_image, - angle, diameter)
            driver.setSteeringAngle(angle)
        else:
            driver.setSteeringAngle(0)

        cv2.imshow('Virtual steering wheel preview', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
