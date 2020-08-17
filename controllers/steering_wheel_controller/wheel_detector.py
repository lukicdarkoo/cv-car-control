import os
import math
import cv2
import numpy as np


def draw_overlay(background_image, foreground_image, x_offset, y_offset):
    """
    Draws an image over an image considering transparency.
    Reference: https://stackoverflow.com/a/14102014/1983050.
    """
    y1 = y_offset
    y2 = y_offset + foreground_image.shape[0]
    x1 = x_offset
    x2 = x_offset + foreground_image.shape[1]
    alpha_foreground = foreground_image[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_foreground
    for c in range(0, 3):
        background_image[y1:y2, x1:x2, c] = \
            alpha_foreground * foreground_image[:, :, c] + \
            alpha_background * background_image[y1:y2, x1:x2, c]


def rotate_image(image, angle):
    """
    Rotates an image for an arbitrary angle (radians).
    Reference: https://stackoverflow.com/a/9042907/1983050.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(
        image_center, angle * 180 / math.pi, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def filter_overlaps(objects):
    """Filters out objects that overlap leaving the object with highest confidence."""
    filtered_objects = []
    for object_1 in objects:
        accept_object_1 = True
        for object_2 in objects:
            if object_1['x'] < object_2['x'] + object_2['width'] and \
                    object_1['x'] + object_1['width'] > object_2['x'] and \
                    object_1['y'] < object_2['y'] + object_2['height'] and \
                    object_1['y'] + object_1['height'] > object_2['y'] and \
                    object_1['confidence'] < object_2['confidence']:
                accept_object_1 = False
        if accept_object_1:
            filtered_objects.append(object_1)
    return filtered_objects


def filter_by_confidence(objects, n_objects=2):
    """Leaves `n_objects` objects with the highest confidence."""
    objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
    return objects[:n_objects]


def get_wheel_properties(objects):
    """Returns rotation and diameter of the virtual steering wheel based on detected fists."""
    left_object = objects[1]
    right_object = objects[0]
    if objects[0]['center_x'] < objects[1]['center_x']:
        left_object = objects[0]
        right_object = objects[1]
    width_diff = right_object['center_x'] - left_object['center_x']
    height_diff = right_object['center_y'] - left_object['center_y']
    angle = math.atan2(height_diff, width_diff)
    diameter = math.sqrt(width_diff**2 + height_diff**2)
    center_x = (right_object['center_x'] + left_object['center_x']) / 2
    center_y = (right_object['center_y'] + left_object['center_y']) / 2
    return angle, center_x, center_y, diameter


def get_objects_from_net(net, output_layers, image, threshold=0):
    """Parses output from the neural net."""
    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    image_height, image_width, _ = image.shape
    objects = []
    for out in outs:
        for detection in out:
            confidence = detection[5]
            if confidence > threshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                objects.append({
                    'x': x,
                    'y': y,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })
    return objects


def draw_wheel(background_image, wheel_image, angle, diameter):
    """Draws steering wheel on the background image and steering wheel properties."""
    diameter = int(min(diameter, background_image.shape[0]))
    y = background_image.shape[0] / 2 - diameter / 2
    x = background_image.shape[1] / 2 - diameter / 2
    wheel_transformed = cv2.resize(wheel_image, (diameter, diameter))
    wheel_transformed = rotate_image(wheel_transformed, angle)
    draw_overlay(background_image, wheel_transformed, int(x), int(y))


def draw_rectangles(objects, image):
    """Draw bounding boxes around the objects."""
    for object in objects:
        x = object['x']
        y = object['y']
        w = object['width']
        h = object['height']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def main():
    controller_dir = os.path.dirname(__file__)
    capture = cv2.VideoCapture(0)

    # Load YOLO
    net = cv2.dnn.readNet(os.path.join(controller_dir, 'yolov3-tiny_train.backup'),
                          os.path.join(controller_dir, 'yolov3-tiny.cfg'))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]

    wheel_image = cv2.imread(os.path.join(
        controller_dir, 'steering_wheel.png'), -1)

    while True:
        _, image = capture.read()
        image = cv2.flip(image, 1)
        objects = get_objects_from_net(net, output_layers, image)
        objects = filter_overlaps(objects)
        objects = filter_by_confidence(objects)

        draw_rectangles(objects, image)

        if len(objects) == 2:
            angle, _, _, diameter = get_wheel_properties(objects)
            draw_wheel(image, wheel_image, - angle, diameter)

        cv2.imshow('Virtual steering wheel preview', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
